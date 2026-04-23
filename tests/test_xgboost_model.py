"""Pytest unit tests for src/models/base_model.py and src/models/xgboost_model.py."""

import os
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.models.base_model import BaseFinanceModel
from src.models.xgboost_model import XGBoostModel

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_classification_data(
    n_train: int = 200,
    n_test: int = 50,
    n_features: int = 7,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    feature_names = [f"f{i}" for i in range(n_features)]
    X_train = pd.DataFrame(rng.standard_normal((n_train, n_features)), columns=feature_names)
    y_train = pd.Series(rng.integers(0, 2, n_train), name="target")
    X_test = pd.DataFrame(rng.standard_normal((n_test, n_features)), columns=feature_names)
    y_test = pd.Series(rng.integers(0, 2, n_test), name="target")
    return X_train, y_train, X_test, y_test


def _minimal_xgb_cfg():
    return OmegaConf.create({
        "name": "xgboost",
        "calibration": {"method": "isotonic", "ensemble": False},
        "optuna": {"n_trials": 2, "direction": "maximize", "timeout_seconds": None},
        "search_space": {
            "n_estimators": {"type": "int", "low": 10, "high": 20},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1}
        },
        "artifacts": {
            "output_dir": "data/processed/models",
            "model_prefix": "xgb",
            "best_params_file": "data/processed/models/xgb_best_params.json",
            "performance_file": "data/processed/models/xgb_performance.json",
        },
    })


def _minimal_xgb_params() -> dict:
    return {
        "n_estimators": 50,
        "learning_rate": 0.1,
        "max_depth": 3,
    }


# ===========================================================================
# BaseFinanceModel Tests
# ===========================================================================

class TestBaseFinanceModelABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            BaseFinanceModel(params={})  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self):
        class ConcreteModel(BaseFinanceModel):
            def fit(self, X_train, y_train, X_val=None, y_val=None):
                self.is_fitted_ = True
                return self
            def predict(self, X):
                self._check_is_fitted()
                return np.zeros(len(X), dtype=int)
            def predict_proba(self, X):
                self._check_is_fitted()
                return np.column_stack([np.ones(len(X)) * 0.5] * 2)
            def tune(self, X_train, y_train, trial):
                return 0.95
            def save(self, path):
                self._check_is_fitted()

        m = ConcreteModel(params={"test": True})
        assert not m.is_fitted_
        m.fit(pd.DataFrame(), pd.Series())
        assert m.is_fitted_

    def test_check_is_fitted_raises_when_unfitted(self):
        class MinModel(BaseFinanceModel):
            def fit(self, X_train, y_train, X_val=None, y_val=None): return self
            def predict(self, X): return np.array([])
            def predict_proba(self, X): return np.array([[]])
            def tune(self, X_train, y_train, trial): return 0.0
            def save(self, path): pass

        m = MinModel(params={})
        with pytest.raises(RuntimeError, match="has not been fitted"):
            m._check_is_fitted()


# ===========================================================================
# XGBoostModel Tests
# ===========================================================================

class TestXGBoostModelFit:
    def test_fit_returns_self(self):
        X_train, y_train, _, _ = _make_classification_data()
        model = XGBoostModel(params=_minimal_xgb_params(), cfg=_minimal_xgb_cfg())
        result = model.fit(X_train, y_train)
        assert result is model

    def test_fit_sets_is_fitted(self):
        X_train, y_train, _, _ = _make_classification_data()
        model = XGBoostModel(params=_minimal_xgb_params(), cfg=_minimal_xgb_cfg())
        model.fit(X_train, y_train)
        assert model.is_fitted_

    def test_fit_populates_calibrated_model(self):
        X_train, y_train, _, _ = _make_classification_data()
        model = XGBoostModel(params=_minimal_xgb_params(), cfg=_minimal_xgb_cfg())
        model.fit(X_train, y_train)
        assert model.calibrated_model_ is not None


class TestXGBoostModelTune:
    def test_tune_evaluates_single_trial(self):
        import optuna
        X_train, y_train, _, _ = _make_classification_data()
        model = XGBoostModel(params={}, cfg=_minimal_xgb_cfg())

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        score = model.tune(X_train, y_train, trial)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestXGBoostModelPredict:
    @pytest.fixture(autouse=True)
    def fitted_model(self):
        X_train, y_train, X_test, y_test = _make_classification_data()
        self.model = XGBoostModel(params=_minimal_xgb_params(), cfg=_minimal_xgb_cfg())
        self.model.fit(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test

    def test_predict_returns_binary_array(self):
        preds = self.model.predict(self.X_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self):
        probs = self.model.predict_proba(self.X_test)
        assert probs.shape == (len(self.X_test), 2)


class TestXGBoostModelSaveLoad:
    def test_save_creates_file(self, tmp_path):
        X_train, y_train, _, _ = _make_classification_data()
        model = XGBoostModel(params=_minimal_xgb_params(), cfg=_minimal_xgb_cfg())
        model.fit(X_train, y_train)
        path = str(tmp_path / "test_model.pkl")
        model.save(path)
        assert os.path.exists(path)

    def test_load_restores_predictions(self, tmp_path):
        X_train, y_train, X_test, _ = _make_classification_data()
        model = XGBoostModel(params=_minimal_xgb_params(), cfg=_minimal_xgb_cfg())
        model.fit(X_train, y_train)
        original_probs = model.predict_proba(X_test)
        path = str(tmp_path / "model.pkl")
        model.save(path)
        loaded: XGBoostModel = XGBoostModel.load(path)  # type: ignore[assignment]
        loaded_probs = loaded.predict_proba(X_test)
        np.testing.assert_array_almost_equal(original_probs, loaded_probs)
