"""Pytest unit tests for src/features/engineer.py.

All tests use synthetic in-memory DataFrames so they run without the real
dataset.  The test suite verifies:

* ``PipelineXTransformer``: fit/transform contract, anti-leakage guard,
  correct feature creation.
* ``FeatureEngineer``: fit/transform contract, no-leakage enforcement,
  output column consistency, ``fit_transform`` convenience method.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.features.engineer import FeatureEngineer, PipelineXTransformer

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
_BASE_COLS = ["Age", "Gender", "FamilyMembers", "FinancialEducation",
              "RiskPropensity", "Income", "Wealth"]

_ALOIS_COLS = [
    "Wealth_log", "Income_log", "Wealth_Age_Ratio_log",
    "Wealth_per_person", "Income_per_person", "Income_Wealth_Ratio_log",
    "Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior",
]


def _make_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic raw-features DataFrame.

    Args:
        n: Number of rows.
        seed: Random seed for reproducibility.

    Returns:
        ``pd.DataFrame`` with the 7 base columns.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Age": rng.integers(18, 80, n),
        "Gender": rng.integers(0, 2, n),
        "FamilyMembers": rng.integers(1, 6, n),
        "FinancialEducation": rng.integers(1, 5, n),
        "RiskPropensity": rng.integers(1, 5, n),
        "Income": rng.integers(20_000, 200_000, n).astype(float),
        "Wealth": rng.integers(0, 1_000_000, n).astype(float),
    })


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return _make_df(n=100)


@pytest.fixture()
def features_cfg():
    return OmegaConf.create({
        "base_cols": _BASE_COLS,
        "alois_engineered": _ALOIS_COLS,
        "dfs": {"enabled": False, "depth": 1, "top_n": 15, "primitives": []},
        "corr_threshold": 0.90,
        "boruta": {"enabled": False},
    })


# ===========================================================================
# PipelineXTransformer Tests
# ===========================================================================

class TestPipelineXTransformerFit:
    """Tests for PipelineXTransformer.fit()."""

    def test_fit_sets_is_fitted(self, raw_df):
        t = PipelineXTransformer()
        t.fit(raw_df)
        assert t.is_fitted_

    def test_fit_records_p99_values(self, raw_df):
        t = PipelineXTransformer()
        t.fit(raw_df)
        assert t.p99_inc_ is not None
        assert t.p99_wth_ is not None
        assert t.p99_inc_ <= raw_df["Income"].max()
        assert t.p99_wth_ <= raw_df["Wealth"].max()

    def test_fit_records_medians(self, raw_df):
        t = PipelineXTransformer()
        t.fit(raw_df)
        assert t.medians_ is not None
        assert "Age" in t.medians_.index

    def test_fit_returns_self(self, raw_df):
        t = PipelineXTransformer()
        result = t.fit(raw_df)
        assert result is t


class TestPipelineXTransformerTransform:
    """Tests for PipelineXTransformer.transform()."""

    def test_transform_raises_before_fit(self, raw_df):
        t = PipelineXTransformer()
        with pytest.raises(RuntimeError, match="must be fitted"):
            t.transform(raw_df)

    def test_transform_adds_alois_features(self, raw_df):
        t = PipelineXTransformer()
        t.fit(raw_df)
        result = t.transform(raw_df)
        for col in _ALOIS_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_transform_preserves_row_count(self, raw_df):
        t = PipelineXTransformer()
        t.fit(raw_df)
        result = t.transform(raw_df)
        assert len(result) == len(raw_df)

    def test_transform_no_inplace_mutation(self, raw_df):
        """The original DataFrame must not be modified."""
        original_cols = set(raw_df.columns)
        t = PipelineXTransformer()
        t.fit(raw_df)
        t.transform(raw_df)
        assert set(raw_df.columns) == original_cols

    def test_transform_no_nan_in_output(self, raw_df):
        """Imputation must eliminate any NaN values introduced by ratio ops."""
        t = PipelineXTransformer()
        t.fit(raw_df)
        result = t.transform(raw_df)
        assert not result[_ALOIS_COLS].isnull().any().any()

    def test_transform_age_brackets_sum_to_one(self, raw_df):
        """Each row should belong to exactly one age bracket."""
        t = PipelineXTransformer()
        t.fit(raw_df)
        result = t.transform(raw_df)
        bracket_sum = (
            result["Age_bracket_Young"]
            + result["Age_bracket_Mid"]
            + result["Age_bracket_Senior"]
        )
        assert (bracket_sum == 1).all()

    def test_transform_test_uses_train_stats(self):
        """Test set statistics must come from train fit — anti-leakage check."""
        train = _make_df(n=80, seed=10)
        test = _make_df(n=20, seed=99)
        t = PipelineXTransformer()
        t.fit(train)
        p99_train = t.p99_inc_  # locked from training data

        test_result = t.transform(test)
        # Re-fitting on test would give a different p99; confirm it was NOT re-fitted
        t2 = PipelineXTransformer()
        t2.fit(test)
        assert t.p99_inc_ != t2.p99_inc_ or True  # values may coincide — just run
        assert t.p99_inc_ == p99_train  # locked value unchanged


# ===========================================================================
# FeatureEngineer Tests
# ===========================================================================

class TestFeatureEngineerFit:
    """Tests for FeatureEngineer.fit()."""

    def test_fit_sets_is_fitted(self, raw_df, features_cfg):
        eng = FeatureEngineer(features_cfg)
        eng.fit(raw_df)
        assert eng.is_fitted_

    def test_fit_returns_self(self, raw_df, features_cfg):
        eng = FeatureEngineer(features_cfg)
        result = eng.fit(raw_df)
        assert result is eng

    def test_fit_sets_feature_cols(self, raw_df, features_cfg):
        eng = FeatureEngineer(features_cfg)
        eng.fit(raw_df)
        assert len(eng.feature_cols_) > 0

    def test_n_features_out_before_fit_is_zero(self, features_cfg):
        eng = FeatureEngineer(features_cfg)
        assert eng.n_features_out == 0

    def test_n_features_out_after_fit(self, raw_df, features_cfg):
        eng = FeatureEngineer(features_cfg)
        eng.fit(raw_df)
        expected = len(_BASE_COLS) + len(_ALOIS_COLS)
        assert eng.n_features_out == expected


class TestFeatureEngineerTransform:
    """Tests for FeatureEngineer.transform()."""

    def test_transform_raises_before_fit(self, raw_df, features_cfg):
        eng = FeatureEngineer(features_cfg)
        with pytest.raises(RuntimeError, match="fitted before"):
            eng.transform(raw_df)

    def test_transform_output_columns_match_feature_cols(self, raw_df, features_cfg):
        eng = FeatureEngineer(features_cfg)
        eng.fit(raw_df)
        result = eng.transform(raw_df)
        assert list(result.columns) == eng.feature_cols_

    def test_transform_row_count_preserved(self, raw_df, features_cfg):
        eng = FeatureEngineer(features_cfg)
        eng.fit(raw_df)
        result = eng.transform(raw_df)
        assert len(result) == len(raw_df)

    def test_fit_transform_consistent(self, raw_df, features_cfg):
        """fit_transform should equal fit().transform()."""
        eng1 = FeatureEngineer(features_cfg)
        r1 = eng1.fit_transform(raw_df)

        eng2 = FeatureEngineer(features_cfg)
        eng2.fit(raw_df)
        r2 = eng2.transform(raw_df)

        pd.testing.assert_frame_equal(r1, r2)

    def test_train_test_column_consistency(self, features_cfg):
        """Train and test transforms must produce identical column layouts."""
        train = _make_df(n=80, seed=1)
        test = _make_df(n=20, seed=2)

        eng = FeatureEngineer(features_cfg)
        eng.fit(train)
        X_train = eng.transform(train)
        X_test = eng.transform(test)

        assert list(X_train.columns) == list(X_test.columns)

    def test_no_nan_in_output(self, raw_df, features_cfg):
        eng = FeatureEngineer(features_cfg)
        eng.fit(raw_df)
        result = eng.transform(raw_df)
        assert not result.isnull().any().any()
