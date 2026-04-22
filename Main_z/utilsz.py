"""
=============================================================================
utilsz.py — DOUBLE-STAGE DATA CONTRACT FOR PIPELINE Z
=============================================================================
PURPOSE:
    The single, authoritative interface for all data access in Pipeline Z.
    Supports two stages with strict caching:
      1. STAGE="base"   : Reads from Dataset_Needs_SOTA.csv (7 raw features)
      2. STAGE="master" : Reads from Master_Needs_SOTA_Z.csv (15 engineered)
=============================================================================
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

PIPELINE_Z_DIR   = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_Z")

# Entrambi i file risiedono nella cartella di output del progetto
BASE_DATA_PATH   = os.path.join(PIPELINE_Z_DIR, "Dataset_Needs_SOTA.csv")
MASTER_DATA_PATH = os.path.join(PIPELINE_Z_DIR, "Master_Needs_SOTA_Z.csv")

# Sorgente originale (Professor files)
RAW_PROFESSOR_PATH = os.path.join(_PROJECT_ROOT, "Dataset2_Needs.xls")

TARGET_COLS  = ["AccumulationInvestment", "IncomeInvestment"]
RANDOM_STATE = 42
FOLD_COL     = "stratified_fold"

# FEATURE_COLS is populated safely on import
FEATURE_COLS: list[str] = []

# Stage-aware caching to prevent I/O disk thrashing
_df_cache = {
    "base": None,
    "master": None
}

def _get_df(stage="master") -> pd.DataFrame:
    """Loads either the raw 7-feat bible or the 15-feat master dataset."""
    global FEATURE_COLS
    
    if stage not in ["base", "master"]:
        raise ValueError(f"[utilsz] Invalid stage '{stage}'. Use 'base' or 'master'.")

    # Return cached version if available
    if _df_cache[stage] is not None:
        return _df_cache[stage]
        
    path_to_load = BASE_DATA_PATH if stage == "base" else MASTER_DATA_PATH

    if not os.path.exists(path_to_load):
        if stage == "master":
            raise FileNotFoundError(f"[utilsz] Master dataset not found at {path_to_load}. Run 01z first.")
        else:
            raise FileNotFoundError(f"[utilsz] Base dataset not found at {path_to_load}. Run 00z first.")
            
    df = pd.read_csv(path_to_load)
    df.columns = df.columns.str.strip()

    # Cache the dataframe
    _df_cache[stage] = df

    # Only update FEATURE_COLS if we are loading the master stage
    # (Since 02z and 03z rely on the master features)
    if stage == "master":
        FEATURE_COLS = [
            c for c in df.columns
            if c not in TARGET_COLS and c not in ["ID", FOLD_COL] and not c.startswith("Unnamed")
        ]
        
    return df

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_full_train_val(stage="master") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns all 4000 Train/Val rows from the requested stage."""
    df = _get_df(stage)
    mask = df[FOLD_COL] >= 0
    
    # Dynamically select feature cols based on what exists in this stage's df
    stage_feats = [c for c in df.columns if c not in TARGET_COLS + ["ID", FOLD_COL]]
    
    X = df[mask][["ID"] + stage_feats].copy()
    y = df[mask][TARGET_COLS].astype(int).copy()
    return X, y

def get_test_set(stage="master") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns the 1000 blind test rows from the requested stage."""
    df = _get_df(stage)
    mask = df[FOLD_COL] == -1
    
    stage_feats = [c for c in df.columns if c not in TARGET_COLS + ["ID", FOLD_COL]]
    
    X = df[mask][["ID"] + stage_feats].copy()
    y = df[mask][TARGET_COLS].astype(int).copy()
    return X, y

def get_train_fold(fold_id: int, stage="master") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns a train/val split for CV using the requested stage."""
    df = _get_df(stage)
    
    if fold_id not in range(5):
        raise ValueError(f"[utilsz] fold_id must be 0-4, got {fold_id}")

    tr_mask  = (df[FOLD_COL] >= 0) & (df[FOLD_COL] != fold_id)
    val_mask = df[FOLD_COL] == fold_id

    stage_feats = [c for c in df.columns if c not in TARGET_COLS + ["ID", FOLD_COL]]

    X_train = df[tr_mask][["ID"] + stage_feats].copy()
    y_train = df[tr_mask][TARGET_COLS].astype(int).copy()
    X_val   = df[val_mask][["ID"] + stage_feats].copy()
    y_val   = df[val_mask][TARGET_COLS].astype(int).copy()

    return X_train, y_train, X_val, y_val

def get_cv_splitter(stage="master") -> list[tuple[list[int], list[int]]]:
    """Returns a list of (train_idx, val_idx) pairs for CalibratedClassifierCV."""
    df = _get_df(stage)
    tv_df = df[df[FOLD_COL] >= 0].reset_index(drop=True)

    splits = []
    for fold_id in range(5):
        val_idx   = tv_df.index[tv_df[FOLD_COL] == fold_id].tolist()
        train_idx = tv_df.index[tv_df[FOLD_COL] != fold_id].tolist()
        splits.append((train_idx, val_idx))
    return splits

def get_raw_professor_data() -> pd.DataFrame:
    """Loads the original raw XLS file from the professor's folder."""
    if not os.path.exists(RAW_PROFESSOR_PATH):
        raise FileNotFoundError(f"[utilsz] Raw professor file not found at {RAW_PROFESSOR_PATH}")
    
    df = pd.read_excel(RAW_PROFESSOR_PATH)
    df.columns = df.columns.str.strip()
    return df

# ---------------------------------------------------------------------------
# PipelineXTransformer — The Anti-Leakage Shield
# ---------------------------------------------------------------------------
class PipelineXTransformer:
    """
    Handles clipping, imputation, and ratio engineering without data leakage.
    FITS only on training data; TRANSFORMS both training and validation/test.
    """
    def __init__(self):
        self.medians = None
        self.p99_inc = None
        self.p99_wth = None
        self.inc_max = None
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame):
        """Calculates stats from the training block only."""
        # Use a copy to avoid side-effects
        df = df_train.copy()
        
        # 1. Quantiles for clipping
        self.p99_inc = df["Income"].quantile(0.99)
        self.p99_wth = df["Wealth"].quantile(0.99)
        self.inc_max = df["Income"].max()
        
        # 2. Medians for imputation
        # Note: We compute medians on the raw features that exist in df
        self.medians = df.median(numeric_only=True)
        
        self.is_fitted = True
        return self

    def transform(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted stats to derive features."""
        if not self.is_fitted:
            raise RuntimeError("[PipelineXTransformer] Must call fit() before transform().")
            
        df = df_in.copy()
        
        # 1. Imputation MUST happen first! (Anti-leakage)
        # If we don't impute Age first, missing ages will generate all-zero dummy brackets.
        df.fillna(self.medians, inplace=True)

        # 2. Age brackets (Non-linear age effect)
        # Now it's safe to cut because there are no NaNs in Age.
        df["Age_bracket"] = pd.cut(
            df["Age"], bins=[17, 35, 55, 100],
            labels=["Young", "Mid", "Senior"]
        )
        dummies = pd.get_dummies(df["Age_bracket"], prefix="Age_bracket", drop_first=False, dtype=int)
        
        # Ensure all columns exist even if a category was entirely missing in the batch
        for label in ["Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior"]:
            if label not in dummies.columns:
                dummies[label] = 0
                
        df = pd.concat([
            df.drop(columns=["Age_bracket"]), 
            dummies[["Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior"]]
        ], axis=1)

        # 3. Internal Clipping (for ratios)
        clipped_inc = df["Income"].clip(upper=self.p99_inc)
        clipped_wth = df["Wealth"].clip(upper=self.p99_wth)
        
        # 4. Ratios (Matching Anima Alois logic)
        df["Wealth_log"] = np.log1p(df["Wealth"])
        df["Income_log"] = np.log1p(df["Income"])
        
        # --- NEW FEATURE: Wealth Accumulation Speed ---
        # Calculate "adult years" (from age 18 onwards, so Age - 17)
        # We use .clip(lower=1) to avoid division by zero.
        adult_years = (df["Age"] - 17).clip(lower=1)
        df["Wealth_Age_Ratio_log"] = np.log1p(clipped_wth / adult_years)
        
        safe_fm = df["FamilyMembers"].replace(0, np.nan).fillna(self.medians.get("FamilyMembers", 1))
        df["Wealth_per_person"] = clipped_wth / safe_fm
        df["Income_per_person"] = clipped_inc / safe_fm
        
        safe_wealth = clipped_wth.replace(0, np.nan)
        raw_ratio = clipped_inc.div(safe_wealth).fillna(self.inc_max)
        df["Income_Wealth_Ratio_log"] = np.log1p(raw_ratio)
        
        return df

    def get_params(self):
        """Returns the fitted parameters for inspection or production export."""
        return {
            "medians": self.medians.to_dict() if self.medians is not None else None,
            "p99_inc": self.p99_inc,
            "p99_wth": self.p99_wth,
            "inc_max": self.inc_max
        }

# Populate FEATURE_COLS correctly on import for downstream scripts!
try:
    _get_df("master")
except Exception:
    pass

if __name__ == "__main__":
    print("=" * 68)
    print("utilsz.py — Double-Stage Contract Test")
    print("=" * 68)
    try:
        X, y = get_full_train_val(stage="base")
        print(f"✅ Stage 'base' OK   : {X.shape[1]-1} features detected.")
    except Exception as e:
        print(f"❌ Stage 'base' Error: {e}")
        
    try:
        X, y = get_full_train_val(stage="master")
        print(f"✅ Stage 'master' OK : {X.shape[1]-1} features detected.")
        print(f"✅ FEATURE_COLS Len  : {len(FEATURE_COLS)}")
    except Exception as e:
        print(f"ℹ️  Stage 'master' Note: {e} (Expected if 01z hasn't run)")
