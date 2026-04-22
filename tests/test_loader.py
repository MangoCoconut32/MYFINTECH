"""Pytest unit tests for src/data/loader.py.

Uses synthetic in-memory DataFrames to avoid any dependency on the real
dataset files. The frozen CSV structure (with a ``stratified_fold`` column)
is mimicked via ``tmp_path`` fixtures so tests are fully self-contained.
"""

import os

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.loader import DataLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_frozen_csv(tmp_path: "Path") -> tuple[str, "DictConfig"]:
    """Create a minimal synthetic frozen CSV and a matching DictConfig.

    Returns:
        A tuple of ``(csv_path, cfg)`` where ``cfg`` is wired to the temp file.
    """
    n_rows = 100
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "ID": np.arange(n_rows),
        "Age": rng.integers(18, 80, n_rows),
        "Gender": rng.integers(0, 2, n_rows),
        "FamilyMembers": rng.integers(1, 6, n_rows),
        "FinancialEducation": rng.integers(1, 5, n_rows),
        "RiskPropensity": rng.integers(1, 5, n_rows),
        "Income": rng.integers(20_000, 200_000, n_rows),
        "Wealth": rng.integers(0, 1_000_000, n_rows),
        "AccumulationInvestment": rng.integers(0, 2, n_rows),
        "IncomeInvestment": rng.integers(0, 2, n_rows),
    })

    # 80 train/val rows (folds 0-4), 20 test rows (fold -1)
    df["stratified_fold"] = -1
    for i in range(80):
        df.at[i, "stratified_fold"] = i % 5

    csv_path = str(tmp_path / "Dataset_Needs_SOTA.csv")
    df.to_csv(csv_path, index=False)

    cfg = OmegaConf.create({
        "dfs_csv_path": csv_path,
        "raw_excel_path": "Dataset2_Needs.xls",
        "frozen_csv_path": csv_path,
        "train_val_size": 80,
        "test_size": 20,
        "n_splits": 5,
        "random_state": 42,
        "id_col": "ID",
        "fold_col": "stratified_fold",
        "target_cols": ["AccumulationInvestment", "IncomeInvestment"],
        "stratify_col": "stratify_combined",
        "primary_target": "IncomeInvestment",
    })
    return csv_path, cfg


# ---------------------------------------------------------------------------
# Tests — DataLoader.load()
# ---------------------------------------------------------------------------

class TestDataLoaderLoad:
    """Tests for the ``load()`` method."""

    def test_load_returns_dataframe(self, synthetic_frozen_csv):
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        df = loader.load()
        assert isinstance(df, pd.DataFrame)

    def test_load_correct_row_count(self, synthetic_frozen_csv):
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        df = loader.load()
        assert len(df) == 100

    def test_load_is_cached(self, synthetic_frozen_csv):
        """Subsequent calls must return the same object without re-reading disk."""
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        df1 = loader.load()
        df2 = loader.load()
        assert df1 is df2

    def test_load_raises_if_file_missing(self, tmp_path):
        cfg = OmegaConf.create({
            "dfs_csv_path": "nonexistent.csv",
            "raw_excel_path": "nonexistent.xls",
            "frozen_csv_path": str(tmp_path / "ghost.csv"),
            "train_val_size": 80,
            "test_size": 20,
            "n_splits": 5,
            "random_state": 42,
            "id_col": "ID",
            "fold_col": "stratified_fold",
            "target_cols": ["AccumulationInvestment", "IncomeInvestment"],
            "stratify_col": "stratify_combined",
            "primary_target": "IncomeInvestment",
        })
        loader = DataLoader(cfg)
        with pytest.raises(FileNotFoundError):
            loader.load()


# ---------------------------------------------------------------------------
# Tests — DataLoader.get_train_val() / get_test_set()
# ---------------------------------------------------------------------------

class TestDataLoaderSplits:
    """Tests for split methods."""

    def test_train_val_size(self, synthetic_frozen_csv):
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        X_tv, y_tv = loader.get_train_val()
        assert len(X_tv) == 80
        assert len(y_tv) == 80

    def test_test_size(self, synthetic_frozen_csv):
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        X_test, y_test = loader.get_test_set()
        assert len(X_test) == 20
        assert len(y_test) == 20

    def test_no_target_leakage_in_X(self, synthetic_frozen_csv):
        """Target columns must not appear in the feature matrix."""
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        X_tv, _ = loader.get_train_val()
        for col in ["AccumulationInvestment", "IncomeInvestment"]:
            assert col not in X_tv.columns

    def test_no_id_or_fold_in_X(self, synthetic_frozen_csv):
        """ID and fold columns must be stripped from X."""
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        X_tv, _ = loader.get_train_val()
        assert "ID" not in X_tv.columns
        assert "stratified_fold" not in X_tv.columns

    def test_y_is_binary(self, synthetic_frozen_csv):
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        _, y_tv = loader.get_train_val(target="IncomeInvestment")
        assert set(y_tv.unique()).issubset({0, 1})

    def test_splits_are_disjoint(self, synthetic_frozen_csv):
        """Train/val and test row indices must be non-overlapping."""
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        df = loader.load()
        tv_ids = set(df[df["stratified_fold"] >= 0]["ID"].tolist())
        test_ids = set(df[df["stratified_fold"] == -1]["ID"].tolist())
        assert tv_ids.isdisjoint(test_ids)


# ---------------------------------------------------------------------------
# Tests — DataLoader.get_fold()
# ---------------------------------------------------------------------------

class TestDataLoaderGetFold:
    """Tests for the per-fold getter."""

    def test_get_fold_valid_id(self, synthetic_frozen_csv):
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        X_train, y_train, X_val, y_val = loader.get_fold(fold_id=0)
        assert len(X_val) > 0
        assert len(X_train) > len(X_val)

    def test_get_fold_invalid_id_raises(self, synthetic_frozen_csv):
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        with pytest.raises(ValueError, match="fold_id must be in"):
            loader.get_fold(fold_id=99)

    def test_get_fold_disjoint_sets(self, synthetic_frozen_csv):
        _, cfg = synthetic_frozen_csv
        loader = DataLoader(cfg)
        X_train, _, X_val, _ = loader.get_fold(fold_id=2)
        # Column sets should be identical (same features)
        assert set(X_train.columns) == set(X_val.columns)
