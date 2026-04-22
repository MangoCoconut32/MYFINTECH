"""MYFINTECH Pipeline — single Hydra-managed entry point.

Run via::

    python main.py pipeline.step=data_prep   # Freeze dataset + engineer features
    python main.py pipeline.step=train       # Train XGBoost (default model)
    python main.py pipeline.step=evaluate    # Load saved model, report metrics
    python main.py pipeline.step=audit       # Placeholder for XAI / fairness (Phase 3)

Override any config value inline::

    python main.py pipeline.step=train model=xgboost model.optuna.n_trials=50
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ===========================================================================
# PipelineOrchestrator
# ===========================================================================
class PipelineOrchestrator:
    """Dispatches execution to the correct pipeline stage based on ``cfg.pipeline.step``.

    Attributes:
        cfg: The fully-composed Hydra ``DictConfig`` for this run.

    Args:
        cfg: Hydra ``DictConfig`` — the top-level composed config.
    """

    _VALID_STEPS: tuple[str, ...] = ("data_prep", "train", "evaluate", "audit")

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg: DictConfig = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the pipeline step specified by ``cfg.pipeline.step``.

        Dispatches to the appropriate private method.

        Raises:
            ValueError: If ``cfg.pipeline.step`` is not one of the valid steps.
        """
        step: str = self.cfg.pipeline.step
        logger.info("PipelineOrchestrator: executing step '%s'.", step)

        if step == "data_prep":
            self._run_data_prep()
        elif step == "train":
            self._run_train()
        elif step == "evaluate":
            self._run_evaluate()
        elif step == "audit":
            self._run_audit()
        else:
            raise ValueError(
                f"Unknown pipeline step: '{step}'. "
                f"Valid options are: {self._VALID_STEPS}"
            )

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _run_data_prep(self) -> None:
        """Step: data_prep — freeze the raw dataset and build the feature matrix.

        Sequence:
            1. ``DatasetFreezer.freeze()`` — produce ``Dataset_Needs_SOTA.csv``
               with stratified fold assignments (idempotent if already exists).
            2. ``DataLoader.load()``        — validate the frozen CSV.
            3. ``FeatureEngineer.fit()``    — fit the anti-leakage transformer
               on the train/val block only.
            4. Save the engineered train/test CSVs to ``data/processed/``.
        """
        import pandas as pd

        from src.data.freezer import DatasetFreezer
        from src.data.loader import DataLoader
        from src.features.engineer import FeatureEngineer

        logger.info("=" * 60)
        logger.info("STEP: data_prep")
        logger.info("=" * 60)

        # 1. Freeze
        freezer = DatasetFreezer(self.cfg.data)
        freezer.freeze(overwrite=False)

        # 2. Load
        loader = DataLoader(self.cfg.data)
        X_tv_raw, y_tv = loader.get_train_val()
        X_test_raw, y_test = loader.get_test_set()
        logger.info(
            "DataLoader: train/val=%d rows | test=%d rows.",
            len(X_tv_raw), len(X_test_raw),
        )

        # 3. Engineer (fit on train only — anti-leakage)
        engineer = FeatureEngineer(self.cfg.features)
        X_tv_eng: pd.DataFrame = engineer.fit_transform(X_tv_raw)
        X_test_eng: pd.DataFrame = engineer.transform(X_test_raw)
        logger.info(
            "FeatureEngineer: %d output features.", engineer.n_features_out
        )

        # 4. Save processed splits
        import os
        out_dir: str = "data/processed"
        os.makedirs(out_dir, exist_ok=True)

        target: str = self.cfg.data.primary_target
        X_tv_eng.assign(**{target: y_tv.values}).to_csv(
            os.path.join(out_dir, "train_engineered.csv"), index=False
        )
        X_test_eng.assign(**{target: y_test.values}).to_csv(
            os.path.join(out_dir, "test_engineered.csv"), index=False
        )
        logger.info("Processed features saved to '%s'.", out_dir)
        logger.info("data_prep complete.")

    def _run_train(self) -> None:
        """Step: train — load the frozen dataset and fit the configured model.

        Currently supports ``cfg.model.name == "xgboost"``.  Future model
        names (``"ebm"``, ``"tabnet"``) will be dispatched here as they are
        added in later phases.
        """
        from omegaconf import OmegaConf

        from src.data.loader import DataLoader
        from src.features.engineer import FeatureEngineer

        logger.info("=" * 60)
        logger.info("STEP: train (model=%s)", self.cfg.model.name)
        logger.info("=" * 60)

        # Load data
        loader = DataLoader(self.cfg.data)
        X_tv_raw, y_tv = loader.get_train_val()
        X_test_raw, y_test = loader.get_test_set()

        # Engineer features
        engineer = FeatureEngineer(self.cfg.features)
        X_tv = engineer.fit_transform(X_tv_raw)
        X_test = engineer.transform(X_test_raw)

        model_name: str = self.cfg.model.name
        if model_name == "xgboost":
            self._train_xgboost(loader, engineer, X_tv, y_tv, X_test, y_test)
        else:
            raise NotImplementedError(
                f"Model '{model_name}' is not yet implemented. "
                "Planned Phase 2+ models: ebm, tabnet, twotower."
            )

    def _train_xgboost(
        self,
        loader: "DataLoader",  # noqa: F821
        engineer: "FeatureEngineer",  # noqa: F821
        X_tv: "pd.DataFrame",  # noqa: F821
        y_tv: "pd.Series",  # noqa: F821
        X_test: "pd.DataFrame",  # noqa: F821
        y_test: "pd.Series",  # noqa: F821
    ) -> None:
        """Train and evaluate the XGBoost model for all configured targets.

        Args:
            loader: Fitted ``DataLoader`` instance (used to build CV splitter).
            engineer: Fitted ``FeatureEngineer`` instance.
            X_tv: Engineered train/val feature matrix.
            y_tv: Train/val target Series.
            X_test: Engineered test feature matrix.
            y_test: Test target Series.
        """
        import json
        import os

        import numpy as np
        from omegaconf import OmegaConf
        from sklearn.metrics import (
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        from src.models.xgboost_model import XGBoostModel

        target_cols: list[str] = list(self.cfg.data.target_cols)
        params: dict = OmegaConf.to_container(
            self.cfg.model.hyperparameters, resolve=True
        )
        out_dir: str = self.cfg.model.artifacts.output_dir
        os.makedirs(out_dir, exist_ok=True)

        performance: dict = {}

        for target in target_cols:
            logger.info("Training XGBoostModel for target: '%s'.", target)
            _, y_tv_target = loader.get_train_val(target=target)
            _, y_test_target = loader.get_test_set(target=target)

            model = XGBoostModel(params=params, cfg=self.cfg.model)

            # Build pre-frozen CV splitter from DataLoader
            cv_splits = self._build_cv_splitter(loader, target)
            model.fit(X_tv, y_tv_target, cv_splits=cv_splits)

            # Evaluate on blind test set
            probs: np.ndarray = model.predict_proba(X_test)[:, 1]
            preds: np.ndarray = model.predict(X_test)

            metrics = {
                "AUC": round(float(roc_auc_score(y_test_target, probs)), 4),
                "Brier": round(float(brier_score_loss(y_test_target, probs)), 4),
                "Precision": round(float(precision_score(y_test_target, preds, zero_division=0)), 4),
                "Recall": round(float(recall_score(y_test_target, preds, zero_division=0)), 4),
                "F1": round(float(f1_score(y_test_target, preds, zero_division=0)), 4),
            }
            performance[target] = metrics
            logger.info("Results for '%s': %s", target, metrics)

            # Save model
            safe_name: str = target.replace("Investment", "").lower()[:3]
            model.save(os.path.join(out_dir, f"xgb_{safe_name}.pkl"))

        # Persist performance JSON
        perf_path: str = self.cfg.model.artifacts.performance_file
        os.makedirs(os.path.dirname(perf_path), exist_ok=True)
        with open(perf_path, "w") as fh:
            json.dump(performance, fh, indent=2)
        logger.info("Performance metrics saved to '%s'.", perf_path)
        logger.info("train step complete.")

    def _run_evaluate(self) -> None:
        """Step: evaluate — load a saved model and report metrics on the test set.

        Currently a scaffold; full implementation follows in Phase 2 after the
        evaluation module (``src/evaluation/``) is built.
        """
        logger.info("=" * 60)
        logger.info("STEP: evaluate (placeholder — Phase 2)")
        logger.info("=" * 60)
        logger.warning(
            "The 'evaluate' step is a Phase 2 deliverable. "
            "It will load a saved model from disk and run the full "
            "metrics suite from src/evaluation/."
        )

    def _run_audit(self) -> None:
        """Step: audit — run XAI and fairness diagnostics (Phase 3 placeholder).

        Will integrate SHAP, LIME, DiCE counterfactuals, and MiFID compliance
        checks from ``src/xai/`` when those modules are implemented.
        """
        logger.info("=" * 60)
        logger.info("STEP: audit (placeholder — Phase 3)")
        logger.info("=" * 60)
        logger.warning(
            "The 'audit' step is a Phase 3 deliverable covering SHAP, LIME, "
            "DiCE counterfactuals, and MiFID II compliance (src/xai/)."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cv_splitter(
        loader: "DataLoader",  # noqa: F821
        target: str,
    ) -> list[tuple[list[int], list[int]]]:
        """Build a list of ``(train_idx, val_idx)`` pairs from frozen folds.

        Args:
            loader: A fitted ``DataLoader`` instance.
            target: Target column name (used to determine ``n_splits``).

        Returns:
            List of 5 ``(train_idx, val_idx)`` integer-index pairs compatible
            with ``CalibratedClassifierCV(cv=...)``.
        """
        df = loader.load()
        fold_col: str = loader.cfg.fold_col
        tv_df = df[df[fold_col] >= 0].reset_index(drop=True)
        splits: list[tuple[list[int], list[int]]] = []
        for fold_id in range(int(loader.cfg.n_splits)):
            val_idx = tv_df.index[tv_df[fold_col] == fold_id].tolist()
            train_idx = tv_df.index[tv_df[fold_col] != fold_id].tolist()
            splits.append((train_idx, val_idx))
        return splits


# ===========================================================================
# Hydra entry point
# ===========================================================================
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-managed pipeline entry point.

    Initialises logging, logs the composed config, and delegates execution
    to :class:`PipelineOrchestrator`.

    Args:
        cfg: Fully-composed Hydra ``DictConfig`` for this run.
    """
    # Initialise logging before anything else so all module loggers inherit
    # the correct level.
    log_level: str = cfg.get("training", {}).get("log_level", "INFO")
    setup_logging(log_level)

    logger.info("Initialising MYFINTECH pipeline.")
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))

    orchestrator = PipelineOrchestrator(cfg)
    orchestrator.run()

    logger.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
