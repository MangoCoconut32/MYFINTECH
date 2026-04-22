"""XGBoostModel — calibrated XGBoost classifier for the MYFINTECH pipeline.

Translates ``OLD/Main_x/02x_xgboost_calibrated.py`` into a fully blueprint-
compliant class that inherits from :class:`~src.models.base_model.BaseFinanceModel`.

Key Design Choices
------------------
* **Hyperparameter injection**: All XGBoost params come from the Hydra config
  dict, so changing ``configs/model/xgboost.yaml`` is sufficient to alter the
  model without touching source code.
* **Calibration**: Uses ``CalibratedClassifierCV`` with ``method="isotonic"``
  and the pre-frozen CV splitter to ensure calibration is leak-free.
* **Optuna integration**: :meth:`tune` runs a study and stores the best params
  back into ``self.params`` before the final :meth:`fit`.
* **Pickle serialisation**: Compatible with the default :meth:`BaseFinanceModel.load`.

Usage::

    from omegaconf import OmegaConf, DictConfig
    from src.models.xgboost_model import XGBoostModel

    params = OmegaConf.to_container(cfg.model.hyperparameters, resolve=True)
    model = XGBoostModel(params=params, cfg=cfg.model)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
"""

import json
import logging
import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.models.base_model import BaseFinanceModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseFinanceModel):
    """Optuna-tuned, isotonically-calibrated XGBoost binary classifier.

    Wraps ``XGBClassifier`` inside ``CalibratedClassifierCV`` and exposes the
    standard :class:`~src.models.base_model.BaseFinanceModel` interface.

    Attributes:
        cfg: Hydra DictConfig sub-tree rooted at ``cfg.model``.
        calibrated_model_: The fitted ``CalibratedClassifierCV`` wrapper.
        best_params_: Hyperparameters used in the final fit (possibly updated
            by :meth:`tune`).

    Args:
        params: Dictionary of XGBoost hyperparameters.  Keys must match the
            ``XGBClassifier`` constructor signature.  Typically obtained via
            ``OmegaConf.to_container(cfg.model.hyperparameters, resolve=True)``.
        cfg: Full ``cfg.model`` DictConfig (for calibration settings, Optuna
            config, and artefact paths).
    """

    def __init__(
        self,
        params: dict[str, Any],
        cfg: Optional["DictConfig"] = None,  # noqa: F821
    ) -> None:
        super().__init__(params)
        self.cfg = cfg
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None
        self.best_params_: dict[str, Any] = dict(params)

    # ------------------------------------------------------------------
    # BaseFinanceModel — required implementations
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cv_splits: Optional[list[tuple[list[int], list[int]]]] = None,
    ) -> "XGBoostModel":
        """Fit the calibrated XGBoost model on the training data.

        Builds a base ``XGBClassifier`` using ``self.best_params_``, then
        wraps it in ``CalibratedClassifierCV`` using ``method="isotonic"``.

        Args:
            X_train: Feature matrix for the full train/val block (4 000 rows).
            y_train: Binary target labels for the train/val block.
            X_val: Unused (kept for API parity with neural model subclasses
                that require a held-out validation set for early stopping).
            y_val: Unused (same reason as ``X_val``).
            cv_splits: Optional list of ``(train_idx, val_idx)`` pairs to pass
                as the ``cv`` argument of ``CalibratedClassifierCV``.  When
                ``None``, the sklearn default 5-fold CV is used.

        Returns:
            ``self`` for method chaining.
        """
        logger.info(
            "XGBoostModel: starting fit on %d samples, %d features.",
            len(X_train), X_train.shape[1],
        )

        base_xgb = self._build_base_estimator()

        calib_method: str = (
            self.cfg.calibration.method if self.cfg is not None else "isotonic"
        )
        calib_ensemble: bool = (
            bool(self.cfg.calibration.ensemble) if self.cfg is not None else False
        )

        calibrated = CalibratedClassifierCV(
            estimator=base_xgb,
            method=calib_method,
            cv=cv_splits if cv_splits is not None else 5,
            ensemble=calib_ensemble,
        )

        calibrated.fit(X_train.values, y_train.values)
        self.calibrated_model_ = calibrated
        self.model_ = calibrated  # satisfies BaseFinanceModel.model_
        self.is_fitted_ = True

        logger.info("XGBoostModel: calibrated fit complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class predictions (threshold = 0.5).

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            1-D integer ``np.ndarray`` of shape ``(n_samples,)`` with values
            in ``{0, 1}``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        probs: np.ndarray = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            2-D float ``np.ndarray`` of shape ``(n_samples, 2)`` where column
            index 1 is the probability of the positive class.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        assert self.calibrated_model_ is not None  # appease type checker
        return self.calibrated_model_.predict_proba(X.values)

    def save(self, path: str) -> None:
        """Serialise the fitted model to a ``.pkl`` file.

        Args:
            path: Destination file path (e.g. ``"data/processed/models/xgb_acc.pkl"``).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        abs_path: str = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("XGBoostModel saved to: %s", abs_path)

    # ------------------------------------------------------------------
    # Optuna hyperparameter tuning
    # ------------------------------------------------------------------

    def tune(
        self,
        X_tv: pd.DataFrame,
        y_tv: pd.Series,
        cv_fold_getter: Any,
        target_name: str,
        n_trials: Optional[int] = None,
        warm_start_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run an Optuna study to find the best XGBoost hyperparameters.

        After the study completes, the best parameters are stored in
        ``self.best_params_`` so a subsequent call to :meth:`fit` uses them
        automatically.

        Args:
            X_tv: Full train/val feature DataFrame (4 000 rows).
            y_tv: Train/val target Series.
            cv_fold_getter: A callable ``(fold_id: int) -> (X_tr, y_tr, X_va, y_va)``
                that returns the pre-frozen fold split for a given fold index.
                Typically wraps :meth:`~src.data.loader.DataLoader.get_fold`.
            target_name: Name of the binary target column being optimised
                (e.g. ``"IncomeInvestment"``).
            n_trials: Number of Optuna trials.  Defaults to
                ``cfg.model.optuna.n_trials`` when a config is available.
            warm_start_path: Path to a JSON file of previously found best
                params (used to enqueue a warm-start trial).

        Returns:
            Dictionary of the best hyperparameters found.

        Raises:
            ImportError: If ``optuna`` is not installed.
        """
        try:
            import optuna  # type: ignore[import]
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as exc:
            raise ImportError(
                "optuna is required for hyperparameter tuning. "
                "Install it with: pip install optuna"
            ) from exc

        resolved_trials: int = n_trials or (
            int(self.cfg.optuna.n_trials) if self.cfg is not None else 20
        )
        search_space: Optional["DictConfig"] = (  # noqa: F821
            self.cfg.search_space if self.cfg is not None else None
        )

        logger.info(
            "XGBoostModel.tune: starting Optuna study (%d trials) for target '%s'.",
            resolved_trials, target_name,
        )

        def objective(trial: "optuna.Trial") -> float:  # noqa: F821
            params = self._sample_params(trial, search_space)
            fold_aucs: list[float] = []
            for fold_id in range(5):
                X_tr, y_tr, X_va, y_va = cv_fold_getter(fold_id)
                mdl = XGBClassifier(**params)
                mdl.fit(X_tr.values, y_tr.values)
                probs = mdl.predict_proba(X_va.values)[:, 1]
                fold_aucs.append(float(roc_auc_score(y_va.values, probs)))
            mean_auc = float(np.mean(fold_aucs))
            logger.debug("Trial %d: mean 5-fold AUC = %.4f", trial.number, mean_auc)
            return mean_auc

        study = optuna.create_study(
            direction="maximize",
            study_name=f"XGB_Opt_{target_name}",
        )

        # Warm start: enqueue previously known best params
        if warm_start_path and os.path.exists(warm_start_path):
            try:
                with open(warm_start_path) as fh:
                    prior: dict = json.load(fh)
                if target_name in prior:
                    study.enqueue_trial(prior[target_name])
                    logger.info("Warm start: enqueued prior best trial for '%s'.", target_name)
            except Exception as exc:
                logger.warning("Could not load warm-start params: %s", exc)

        study.optimize(objective, n_trials=resolved_trials, show_progress_bar=False)

        best: dict[str, Any] = study.best_params
        best_auc: float = study.best_value
        logger.info(
            "XGBoostModel.tune: best 5-fold AUC = %.4f for '%s'.",
            best_auc, target_name,
        )
        logger.debug("Best params: %s", best)

        self.best_params_ = {**best, **self._fixed_params()}
        return self.best_params_

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_base_estimator(self) -> XGBClassifier:
        """Construct an ``XGBClassifier`` from ``self.best_params_``.

        Returns:
            An unfitted ``XGBClassifier`` instance.
        """
        # Merge user params with mandatory fixed params (eval_metric, etc.)
        final_params: dict[str, Any] = {**self.best_params_, **self._fixed_params()}
        return XGBClassifier(**final_params)

    @staticmethod
    def _fixed_params() -> dict[str, Any]:
        """Return hyperparameters that are always forced regardless of Optuna.

        Returns:
            Dictionary of non-tunable XGBoost settings.
        """
        return {
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
        }

    @staticmethod
    def _sample_params(
        trial: Any,
        search_space: Optional[Any],
    ) -> dict[str, Any]:
        """Sample a hyperparameter configuration from an Optuna trial.

        When ``search_space`` is provided (Hydra DictConfig), each param's
        type and bounds are read from config.  Falls back to hardcoded
        defaults from ``02x_xgboost_calibrated.py`` when config is absent.

        Args:
            trial: An ``optuna.Trial`` object.
            search_space: Optional Hydra config sub-tree (``cfg.model.search_space``).

        Returns:
            Dictionary of sampled hyperparameters suitable for ``XGBClassifier``.
        """
        if search_space is None:
            # Hardcoded fallback matching the original 02x search space
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }

        # Config-driven sampling
        params: dict[str, Any] = {}
        for param_name, spec in search_space.items():
            param_type: str = spec["type"]
            low = spec["low"]
            high = spec["high"]
            if param_type == "int":
                step = spec.get("step", 1)
                params[param_name] = trial.suggest_int(param_name, int(low), int(high), step=int(step))
            elif param_type == "float":
                log_scale: bool = bool(spec.get("log", False))
                params[param_name] = trial.suggest_float(
                    param_name, float(low), float(high), log=log_scale
                )
        return params
