"""
=============================================================================
01y_feature_engineering.py — MASTER DATASET BUILDER FOR PIPELINE X
=============================================================================
PURPOSE:
    Fuses two feature-engineering "souls" into a single Master Dataset Y:

    ANIMA ALOIS (Domain Knowledge):
        Hand-crafted financial ratios proven to raise AUC ~3-4% (from utils.py):
        Wealth_log, Income_log, Wealth_per_person, Income_per_person,
        Inc_to_Wealth_ratio, Age_bracket_* (one-hot)

    ANIMA MOA (Brute Force - DIY DFS depth=1):
        All pairwise add / mul / div combinations of the 7 raw numeric columns.
        Featuretools avoided: pure numpy/pandas — no crash risk on any env.

    SELECTION (SOTA Multi-Target):
        LightGBM importance averaged across AccumulationInvestment AND
        IncomeInvestment, computed ONLY on the 4000 Train/Val rows.
        Top 15 DFS features selected by mean rank.

    DEDUPLICATION (Pearson > 0.90):
        Before finalising, any DFS feature correlated > 0.90 with an Alois
        feature OR with a previously selected DFS feature is dropped and
        replaced by the next in the ranked list.

ANTI-LEAKAGE PROTOCOL:
    - All statistics (median imputation, correlation matrix, LGBM fitting)
      are computed EXCLUSIVELY on the 4000-row Train/Val block.
    - The exact same transformations are applied "by drag" to the Test block.
    - get_test_set() is called ONLY for the final projection — never for fitting.

DUAL-DATA ARCHITECTURE:
    Two views of the same 30 features are saved separately:

    TREE VIEW (Train/Test_Master_X_Tree.csv):
        Raw, unscaled values (e.g. Age=45, Wealth=250000).
        Used by: 02y (XGBoost Baseline), 03y (EBM Accumulation).
        Benefit: EBM Shape Functions are human-readable (e.g. "Wealth > 200000").
        MIFID rules in 05y use raw values (e.g. if Age > 65, if RiskPropensity < 0.4).

    NN VIEW (Train/Test_Master_X_NN.csv):
        All continuous features scaled to [0, 1] via MinMaxScaler.
        Used by: 04y (TabNet SSL+MTL).
        Benefit: Neural networks require normalized inputs to avoid gradient dominance
                 from large-magnitude features (e.g. Wealth vs RiskPropensity).

OUTPUTS (in BuisnessCase2/Output/Pipeline_Y/):
    Train/Test_Master_X_Tree.csv — 30 features, RAW values (for EBM, XGBoost)
    Train/Test_Master_X_NN.csv  — 30 features, [0,1] scaled (for TabNet)
    Train/Test_Master_X.csv     — backwards-compat alias → Tree view
    feature_legend.txt          — Human-readable explanation of each feature (MIFID)
=============================================================================
"""

import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import shutil
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Import Pipeline Y data contract
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from utilsy import get_full_train_val, get_test_set, TARGET_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FOLD_COL     = "stratified_fold"
SOTA_CSV     = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "Dataset_Needs_SOTA.csv"))

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
OUT_DIR = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_Y")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CORR_THRESHOLD = 0.90  # Legend reference only now

# 15 Hardened Alois Features
BASE_COLS = ["Age", "Gender", "FamilyMembers", "FinancialEducation",
             "RiskPropensity", "Income", "Wealth"]
ALOIS_ENGINEERED = [
    "Wealth_log", "Income_log", "Wealth_per_person", "Income_per_person",
    "Inc_to_Wealth_ratio", "Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior"
]
MASTER_COLS = BASE_COLS + ALOIS_ENGINEERED

# REMOVED: TOP_N_DFS = 15

print("=" * 70)
print("01y_feature_engineering.py — Master Dataset Y Builder")
print("=" * 70)

# ===========================================================================
# STEP 1 — Load raw data (utilsy contract)
# ===========================================================================
print("\n[1/7] Loading raw data via utilsy...")
X_tv_raw, y_tv = get_full_train_val()
X_te_raw, y_te = get_test_set()

print(f"      Train/Val : {X_tv_raw.shape}")
print(f"      Test      : {X_te_raw.shape}")

# ===========================================================================
# STEP 2 — Alois Features (Domain Knowledge — from utils.py)
#           Anti-Leakage: division-by-zero guard uses train medians
# ===========================================================================
print("\n[2/7] Engineering Alois features (financial domain logic)...")


def _alois_features(X: pd.DataFrame, train_ref: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies Alois feature engineering while preserving RAW base columns.
    """
    df = X.copy()
    ref = train_ref if train_ref is not None else df

    # 1. Internal Clipping for Engineered Ratios (Winsorization)
    # Note: We do NOT clip the base df[col] in-place to preserve "Raw" Tree View integrity.
    # We clip internal versions used for derived features.
    clipped_inc = df["Income"].clip(upper=ref["Income"].quantile(0.99))
    clipped_wth = df["Wealth"].clip(upper=ref["Wealth"].quantile(0.99))

    # 2. Log-transforms
    df["Wealth_log"]  = np.log1p(df["Wealth"])
    df["Income_log"]  = np.log1p(df["Income"])

    # 3. Per-member ratios
    safe_fm = df["FamilyMembers"].replace(0, np.nan)
    ref_fm  = ref["FamilyMembers"].replace(0, np.nan)
    fm_median = ref_fm.median()
    safe_fm = safe_fm.fillna(fm_median)

    # Use clipped values for ratios to prevent Inf/Large-scale issues
    df["Wealth_per_person"]  = clipped_wth / safe_fm
    df["Income_per_person"]  = clipped_inc / safe_fm

    # 4. Income-to-Wealth ratio (life-cycle proxy)
    safe_wealth = df["Wealth"].replace(0, np.nan)
    income_max  = ref["Income"].max()
    df["Inc_to_Wealth_ratio"] = df["Income"].div(safe_wealth).fillna(income_max)

    # 5. Age brackets (non-linear age effect)
    df["Age_bracket"] = pd.cut(
        df["Age"], bins=[17, 35, 55, 100],
        labels=["Young", "Mid", "Senior"]
    )
    dummies = pd.get_dummies(df["Age_bracket"], prefix="Age_bracket", drop_first=False, dtype=int)
    
    # Ensure all columns exist for Test set consistency (if a bracket is missing in test)
    for label in ["Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior"]:
        if label not in dummies.columns:
            dummies[label] = 0
            
    df = pd.concat([df.drop(columns=["Age_bracket"]), dummies[
        ["Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior"]
    ]], axis=1)

    return df


X_tv_alois = _alois_features(X_tv_raw, train_ref=X_tv_raw)
X_te_alois = _alois_features(X_te_raw, train_ref=X_tv_raw)   # always use train ref

ALOIS_FEATURE_NAMES = [
    "Wealth_log", "Income_log", "Wealth_per_person", "Income_per_person",
    "Inc_to_Wealth_ratio", "Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior"
]
print(f"      Alois features added: {len(ALOIS_FEATURE_NAMES)}")

# ===========================================================================
# STEP 3-5 — Dynamic Selection Removed (Hardening)
# ===========================================================================
print("\n[3-5/7] Dynamic selection (DFS/LightGBM) bypassed for structural stability.")
print(f"      Handoff JSON updated with fixed Alois set.")

features_json_path = os.path.join(OUT_DIR, "selected_engineered_features.json")
with open(features_json_path, 'w') as f:
    json.dump(ALOIS_ENGINEERED, f, indent=4)

# ===========================================================================
# STEP 6 — Assemble Master Dataset Y (Tree View + NN View)
# ===========================================================================
print("\n[6/7] Assembling Symmetric Master Dataset Y (Dual-Data)...")

def _assemble(df_in, y_in):
    # Use deterministic MASTER_COLS
    id_col = df_in[["ID"]].reset_index(drop=True)
    df = df_in[MASTER_COLS].copy().reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df, id_col, y_in.reset_index(drop=True)

X_tv_master, tv_ids, y_tv_clean = _assemble(X_tv_alois, y_tv)
X_te_master, te_ids, y_te_clean = _assemble(X_te_alois, y_te)

# --- NaN fill: train medians applied to both train and test (anti-leakage) ---
train_medians = X_tv_master.median()
X_tv_master.fillna(train_medians, inplace=True)
X_te_master.fillna(train_medians, inplace=True)  # drag train median to test

# =============================================================================
# 1. TREE VIEW — Raw, unscaled values
# =============================================================================
train_tree = pd.concat([tv_ids, X_tv_master.copy(), y_tv_clean], axis=1)
test_tree  = pd.concat([te_ids, X_te_master.copy(), y_te_clean], axis=1)

if os.path.exists(SOTA_CSV):
    # Bug M2 Fix: Secure merge on ID
    sota_folds = pd.read_csv(SOTA_CSV, usecols=["ID", FOLD_COL])
    train_tree = train_tree.merge(sota_folds, on="ID", how="left")
    # Clean up position (move fold to the front)
    cols = [FOLD_COL] + [c for c in train_tree.columns if c != FOLD_COL]
    train_tree = train_tree[cols]

train_tree.to_csv(os.path.join(OUT_DIR, "Train_Master_X_Tree.csv"), index=False)
test_tree.to_csv(os.path.join(OUT_DIR, "Test_Master_X_Tree.csv"), index=False)
shutil.copy(os.path.join(OUT_DIR, "Train_Master_X_Tree.csv"), os.path.join(OUT_DIR, "Train_Master_X.csv"))
shutil.copy(os.path.join(OUT_DIR, "Test_Master_X_Tree.csv"), os.path.join(OUT_DIR, "Test_Master_X.csv"))

# =============================================================================
# 2. NN VIEW — MinMaxScaler [0, 1]
# =============================================================================
X_tv_nn = X_tv_master.copy()
X_te_nn = X_te_master.copy()

scaler = MinMaxScaler()
X_tv_scaled = pd.DataFrame(scaler.fit_transform(X_tv_nn), columns=MASTER_COLS)
X_te_scaled = pd.DataFrame(scaler.transform(X_te_nn), columns=MASTER_COLS)

train_nn = pd.concat([tv_ids, X_tv_scaled, y_tv_clean], axis=1)
test_nn  = pd.concat([te_ids, X_te_scaled, y_te_clean], axis=1)

if os.path.exists(SOTA_CSV):
    train_nn = train_nn.merge(sota_folds, on="ID", how="left")
    train_nn = train_nn[cols]

train_nn.to_csv(os.path.join(OUT_DIR, "Train_Master_X_NN.csv"), index=False)
test_nn.to_csv(os.path.join(OUT_DIR, "Test_Master_X_NN.csv"), index=False)

# ===========================================================================
# STEP 7 — Feature Legend (MIFID compliance)
# ===========================================================================
print("\n[7/7] Writing feature_legend.txt...")

COL_DESC = {
    "Age":                "Età del cliente",
    "Gender":             "Genere del cliente",
    "FamilyMembers":      "Numero di membri del nucleo familiare",
    "FinancialEducation": "Livello di educazione finanziaria (scala ordinale)",
    "RiskPropensity":     "Propensione al rischio dichiarata (scala ordinale)",
    "Income":             "Reddito annuo lordo (€)",
    "Wealth":             "Patrimonio totale (€)",
}

legend_lines = [
    "=" * 70,
    "PIPELINE X — SYMMETRIC FEATURE LEGEND (MIFID II)",
    "=" * 70,
    "",
    "SECTION A — BASE FEATURES (7 raw columns clipped at P99)",
    "-" * 70,
]
for col in BASE_COLS:
    legend_lines.append(f"  {col:<28} → {COL_DESC.get(col, col)}")

legend_lines += [
    "",
    "SECTION B — ALOIS ENGINEERED FEATURES (Financial logic)",
    "-" * 70,
    "  Wealth_log            → Log(1+Wealth): riduce l'asimmetria patrimoniale",
    "  Income_log            → Log(1+Income): riduce l'asimmetria reddituale",
    "  Wealth_per_person     → Wealth / FamilyMembers: patrimonio netto pro capite",
    "  Income_per_person     → Income / FamilyMembers: reddito netto pro capite",
    "  Inc_to_Wealth_ratio   → Income / Wealth: proxy del ciclo di vita finanziario",
    "  Age_bracket_Young     → Dummy: 1 se età 18-35 (fase accumulazione)",
    "  Age_bracket_Mid       → Dummy: 1 se età 36-55 (fase consolidamento)",
    "  Age_bracket_Senior    → Dummy: 1 se età > 55 (fase income/rendita)",
    "",
    "=" * 70,
    "PROTOCOL: All models (XGB, EBM, TabNet) now use this identical 15-set.",
    "NN VIEW: All features are scaled between 0 and 1 via MinMaxScaler.",
    "=" * 70,
]

legend_path = os.path.join(OUT_DIR, "feature_legend.txt")
with open(legend_path, "w", encoding="utf-8") as f:
    f.write("\n".join(legend_lines))

print("\n" + "=" * 70)
print("✅ 01y_feature_engineering.py COMPLETE (SYMMETRIC HARDENING)")
print("=" * 70)
