"""LightGBMModel implementation for the MYFINTECH pipeline."""

import logging
import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.models.base_model import BaseFinanceModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseFinanceModel):
    """Calibrated LightGBM classifier with internal 5-fold CV tuning."""

    def __init__(self, params: dict[str, Any], cfg: Optional[Any] = None) -> None:
        super().__init__(params)
        self.cfg = cfg
        self.calibrated_model_: Optional[CalibratedClassifierCV] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMModel":
        logger.info("LightGBMModel: fitting on %d samples.", len(X_train))
        base_lgb = LGBMClassifier(**{**self.params, **self._fixed_params()})

        calib_method = self.cfg.calibration.method if self.cfg else "isotonic"
        calib_ensemble = self.cfg.calibration.ensemble if self.cfg else False

        calibrated = CalibratedClassifierCV(
            estimator=base_lgb,
            method=calib_method,
            cv=5,
            ensemble=calib_ensemble,
        )

        calibrated.fit(X_train.values, y_train.values)
        self.calibrated_model_ = calibrated
        self.model_ = calibrated
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self.calibrated_model_.predict_proba(X.values)

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, trial: Any) -> float:
        search_space = self.cfg.search_space if self.cfg else None
        params = self._sample_params(trial, search_space)
        params.update(self._fixed_params())

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []

        X_values = X_train.values
        y_values = y_train.values

        for train_idx, val_idx in skf.split(X_values, y_values):
            X_tr, X_va = X_values[train_idx], X_values[val_idx]
            y_tr, y_va = y_values[train_idx], y_values[val_idx]

            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, probs))

        return float(np.mean(aucs))

    def save(self, path: str) -> None:
        self._check_is_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("LightGBMModel saved to %s", path)

    @staticmethod
    def _fixed_params() -> dict[str, Any]:
        return {"n_jobs": -1, "verbosity": -1}

    @staticmethod
    def _sample_params(trial: Any, search_space: Optional[Any]) -> dict[str, Any]:
        params = {}
        if search_space:
            for name, spec in search_space.items():
                if spec["type"] == "int":
                    params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
                elif spec["type"] == "float":
                    params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        return params
