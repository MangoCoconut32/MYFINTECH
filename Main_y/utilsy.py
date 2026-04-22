"""
=============================================================================
utilsy.py — DATA CONTRACT FOR PIPELINE X
=============================================================================
PURPOSE:
    The single, authoritative interface for all data access in Pipeline Y.
    Reads EXCLUSIVELY from Dataset_Needs_SOTA.csv — the frozen dataset
    produced by 00y_freeze_dataset.py.

    No script in Pipeline Y ever reads the raw Excel file directly.

API:
    get_train_fold(fold_id)  →  X_tr, y_tr, X_val, y_val
    get_test_set()           →  X_test, y_test
    get_full_train_val()     →  X_tv, y_tv   (all 4000 rows, for final refit)

CONSTANTS AVAILABLE FOR IMPORT:
    TARGET_COLS    → ["AccumulationInvestment", "IncomeInvestment"]
    FEATURE_COLS   → list of raw feature columns (no ID, no targets, no fold col)
    DATA_PATH      → absolute path to Dataset_Needs_SOTA.csv
    RANDOM_STATE   → 42

ANTI-LEAKAGE RULE:
    get_test_set() must NEVER be called inside any fitting, CV, or Optuna loop.
    Only 05y_production_engine.py and audit scripts may call it.
=============================================================================
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

DATA_PATH    = os.path.join(_PROJECT_ROOT, "Dataset_Needs_SOTA.csv")
TARGET_COLS  = ["AccumulationInvestment", "IncomeInvestment"]
RANDOM_STATE = 42
FOLD_COL     = "stratified_fold"

# FEATURE_COLS is populated on first import (see _get_df())
FEATURE_COLS: list[str] = []


# ---------------------------------------------------------------------------
# Internal loader (cached in module scope for efficiency)
# ---------------------------------------------------------------------------
_df_cache: pd.DataFrame | None = None


def _get_df() -> pd.DataFrame:
    """
    Loads and caches Dataset_Needs_SOTA.csv.
    Populates the module-level FEATURE_COLS constant on first call.

    Raises
    ------
    FileNotFoundError
        If Dataset_Needs_SOTA.csv does not exist.
        Solution: run 00y_freeze_dataset.py first.
    """
    global _df_cache, FEATURE_COLS

    if _df_cache is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"[utilsy] Frozen dataset not found:\n  {DATA_PATH}\n"
                "Run 00y_freeze_dataset.py first to generate it."
            )
        _df_cache = pd.read_csv(DATA_PATH)
        _df_cache.columns = _df_cache.columns.str.strip()

        # Derive FEATURE_COLS dynamically so this file never needs editing
        # when features are added in 01y_feature_engineering.py
        FEATURE_COLS = [
            c for c in _df_cache.columns
            if c not in TARGET_COLS
            and c not in ["ID", FOLD_COL]
            and not c.startswith("Unnamed")
        ]

    return _df_cache


def _split_X_y(block: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a block into features (includes ID for mapping) and target columns."""
    _get_df()   # ensure FEATURE_COLS is populated
    # We include ID in the dataframe X so engineering scripts can use it for joins.
    # However, FEATURE_COLS (the constant) intentionally EXCLUDES ID to prevent leakage in models.
    X = block[["ID"] + FEATURE_COLS].copy()
    y = block[TARGET_COLS].astype(int).copy()
    return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_train_fold(fold_id: int) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Returns a train/validation split for one inner fold of the CV loop.

    The training set is ALL Train/Val rows EXCEPT fold_id.
    The validation set is ONLY fold_id rows.

    Parameters
    ----------
    fold_id : int
        Integer in [0, 4]. The fold to use as validation.

    Returns
    -------
    X_train : pd.DataFrame, shape (~3200, n_features)
    y_train : pd.DataFrame, shape (~3200, 2)
    X_val   : pd.DataFrame, shape (~800, n_features)
    y_val   : pd.DataFrame, shape (~800, 2)

    Example
    -------
    for fold_id in range(5):
        X_tr, y_tr, X_val, y_val = get_train_fold(fold_id)
        model.fit(X_tr, y_tr['IncomeInvestment'])
        auc = roc_auc_score(y_val['IncomeInvestment'], model.predict_proba(X_val)[:, 1])
    """
    df = _get_df()

    if fold_id not in range(5):
        raise ValueError(f"[utilsy] fold_id must be 0–4, got {fold_id}")

    # Training = all Train/Val rows that are NOT the current validation fold
    tr_mask  = (df[FOLD_COL] >= 0) & (df[FOLD_COL] != fold_id)
    val_mask = df[FOLD_COL] == fold_id

    X_train, y_train = _split_X_y(df[tr_mask])
    X_val,   y_val   = _split_X_y(df[val_mask])

    return X_train, y_train, X_val, y_val


def get_test_set() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns the blind hold-out test set (stratified_fold == -1).

    ⚠️  ANTI-LEAKAGE RULE:
        Call this function ONLY from:
          - 05y_production_engine.py  (final reporting)
          - Audit / backtesting scripts
        NEVER inside a training, CV, or Optuna loop.

    Returns
    -------
    X_test : pd.DataFrame, shape (1000, n_features)
    y_test : pd.DataFrame, shape (1000, 2)
    """
    df = _get_df()
    mask = df[FOLD_COL] == -1
    return _split_X_y(df[mask])


def get_full_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns all 4000 Train/Val rows (folds 0–4 combined).

    Use this for the FINAL REFIT after hyperparameter search is complete.
    Do NOT use this inside a CV loop — use get_train_fold() there.

    Returns
    -------
    X_tv : pd.DataFrame, shape (4000, n_features)
    y_tv : pd.DataFrame, shape (4000, 2)
    """
    df = _get_df()
    mask = df[FOLD_COL] >= 0
    return _split_X_y(df[mask])


def get_cv_splitter() -> list[tuple[list[int], list[int]]]:
    """
    Returns a list of (train_idx, val_idx) pairs for CalibratedClassifierCV.
    Indices are relative to the Train/Val block (4000 rows, folds 0-4).

    Example
    -------
    calibrated = CalibratedClassifierCV(base_model, cv=get_cv_splitter())
    calibrated.fit(X_tv, y_tv)
    """
    df = _get_df()
    # Filter only train/val rows and reset index to ensure indices match the 4000-row X_tv
    tv_df = df[df[FOLD_COL] >= 0].reset_index(drop=True)

    splits = []
    for fold_id in range(5):
        val_idx   = tv_df.index[tv_df[FOLD_COL] == fold_id].tolist()
        train_idx = tv_df.index[tv_df[FOLD_COL] != fold_id].tolist()
        splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# Smoke test (run this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 68)
    print("utilsy.py — Pipeline Y Data Contract Smoke Test")
    print("=" * 68)

    # Test 1: test set size
    X_te, y_te = get_test_set()
    assert len(X_te) == 1000, f"❌ Test set has {len(X_te)} rows, expected 1000"
    print(f"\n✅ Test set size     : {len(X_te)} rows — OK")

    # Test 2: train/val total size
    X_tv, y_tv = get_full_train_val()
    assert len(X_tv) == 4000, f"❌ Train/Val has {len(X_tv)} rows, expected 4000"
    print(f"✅ Train/Val size    : {len(X_tv)} rows — OK")

    # Test 3: 5-fold coverage = all 4000 rows
    total_fold_rows = sum(
        len(get_train_fold(i)[2]) for i in range(5)   # val sizes
    )
    assert total_fold_rows == 4000, \
        f"❌ Sum of val-fold rows = {total_fold_rows}, expected 4000"
    print(f"✅ 5-fold coverage   : {total_fold_rows} val rows total — OK")

    # Test 4: compare Income rate Fold 0 vs Test Set
    _, _, _, y_val0 = get_train_fold(0)
    rate_fold0 = y_val0["IncomeInvestment"].mean()
    rate_test  = y_te["IncomeInvestment"].mean()
    delta      = abs(rate_fold0 - rate_test)
    print(f"\n  Income rate Fold 0 : {rate_fold0:.3f}")
    print(f"  Income rate Test   : {rate_test:.3f}")
    print(f"  Δ                  : {delta:.4f}  {'✅ OK' if delta < 0.05 else '⚠️  Large skew'}")

    # Test 5: no feature leakage — test rows not in train/val
    X_tv, _ = get_full_train_val()
    X_te, _ = get_test_set()
    # Use _df_cache to check IDs since X no longer has them
    df = _get_df()
    tv_ids = df[df[FOLD_COL] >= 0]["ID"]
    te_ids = df[df[FOLD_COL] == -1]["ID"]
    assert not tv_ids.isin(te_ids).any(), "❌ ID overlap detected between Train/Val and Test!"
    print(f"\n✅ No ID overlap      : Train/Val ∩ Test = ∅ — OK")

    # Test 6: CV splitter
    splits = get_cv_splitter()
    assert len(splits) == 5, f"❌ Expected 5 splits, got {len(splits)}"
    for i, (tr, va) in enumerate(splits):
        assert len(tr) + len(va) == 4000, f"❌ Split {i} size mismatch"
        assert len(va) == 800, f"❌ Split {i} validation size mismatch ({len(va)})"
    print(f"✅ CV Splitter        : 5 pairs of (3200 tr, 800 va) — OK")

    print(f"\n  Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"  Target  columns   : {TARGET_COLS}")

    print("\n" + "=" * 68)
    print("✅ All smoke tests passed. utilsy.py is ready for Pipeline Y.")
    print("=" * 68)

# Populate constants on import (Bug M4 fix)
try:
    _get_df()
except:
    pass
