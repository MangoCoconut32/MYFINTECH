"""
=============================================================================
01x_feature_engineering.py — MASTER DATASET BUILDER FOR PIPELINE X
=============================================================================
PURPOSE:
    Fuses two feature-engineering "souls" into a single Master Dataset X:

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
        Used by: 02x (XGBoost Baseline), 03x (EBM Accumulation).
        Benefit: EBM Shape Functions are human-readable (e.g. "Wealth > 200000").
        MIFID rules in 05x use raw values (e.g. if Age > 65, if RiskPropensity < 0.4).

    NN VIEW (Train/Test_Master_X_NN.csv):
        All continuous features scaled to [0, 1] via MinMaxScaler.
        Used by: 04x (TabNet SSL+MTL).
        Benefit: Neural networks require normalized inputs to avoid gradient dominance
                 from large-magnitude features (e.g. Wealth vs RiskPropensity).

OUTPUTS (in BuisnessCase2/Output/Pipeline_X/):
    Train/Test_Master_X_Tree.csv — 30 features, RAW values (for EBM, XGBoost)
    Train/Test_Master_X_NN.csv  — 30 features, [0,1] scaled (for TabNet)
    Train/Test_Master_X.csv     — backwards-compat alias → Tree view
    feature_legend.txt          — Human-readable explanation of each feature (MIFID)
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shutil
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Import Pipeline X data contract
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from utilsx import get_full_train_val, get_test_set, TARGET_COLS, PipelineXTransformer

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
OUT_DIR      = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_X")
SOTA_CSV     = os.path.join(OUT_DIR, "Dataset_Needs_SOTA.csv")
FOLD_COL     = "stratified_fold"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CORR_THRESHOLD = 0.90  # Legend reference only now

# 15 Hardened Alois Features
BASE_COLS = ["Age", "Gender", "FamilyMembers", "FinancialEducation",
             "RiskPropensity", "Income", "Wealth"]
ALOIS_ENGINEERED = [
    "Wealth_log", "Income_log", "Wealth_Age_Ratio_log", "Wealth_per_person", "Income_per_person",
    "Income_Wealth_Ratio_log", "Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior"
]
MASTER_COLS = BASE_COLS + ALOIS_ENGINEERED

# REMOVED: TOP_N_DFS = 15

print("=" * 70)
print("01x_feature_engineering.py — Master Dataset X Builder")
print("=" * 70)

# ===========================================================================
# STEP 1 — Load raw data (utilsx contract - BASE STAGE)
# ===========================================================================
print("\n[1/7] Loading raw data via utilsx (BASE STAGE)...")
X_tv_raw, y_tv = get_full_train_val(stage="base")
X_te_raw, y_te = get_test_set(stage="base")

# X contains 'ID' and the base features. 
# We need to compute statistics ONLY on the Train/Val block (4000 rows).
X_all_raw = pd.concat([X_tv_raw, X_te_raw], axis=0).reset_index(drop=True)
y_all     = pd.concat([y_tv, y_te], axis=0).reset_index(drop=True)

# --- ZONA NEUTRA: PREPARAZIONE FOLD (Bug 6 Fix) ---
sota_folds = None
if os.path.exists(SOTA_CSV):
    print(f"  --> Trovato SOTA_CSV, caricamento fold in memoria...")
    sota_folds = pd.read_csv(SOTA_CSV, usecols=["ID", FOLD_COL])

# ===========================================================================
# STEP 2 — Alois Features (Domain Knowledge — from utils.py)
#           Anti-Leakage: division-by-zero guard uses train medians
# ===========================================================================
print("\n[2/7] Engineering Alois features (financial domain logic)...")


def _alois_features(X: pd.DataFrame, train_ref: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies Alois feature engineering while preserving RAW base columns.
    Uses the hardened PipelineXTransformer logic (Age brackets + Ratios).
    """
    transformer = PipelineXTransformer()
    transformer.fit(train_ref if train_ref is not None else X)
    
    # Apply Transformer (Anti-leakage logic)
    df = transformer.transform(X)
    
    return df


X_tv_alois = _alois_features(X_tv_raw, train_ref=X_tv_raw)
X_te_alois = _alois_features(X_te_raw, train_ref=X_tv_raw)   # always use train ref

ALOIS_FEATURE_NAMES = [
    "Wealth_log", "Income_log", "Wealth_Age_Ratio_log", "Wealth_per_person", "Income_per_person",
    "Income_Wealth_Ratio_log", "Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior"
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
# STEP 6 — Assemble Master Dataset X (Tree View + NN View)
# ===========================================================================
print("\n[6/7] Assembling Symmetric Master Dataset X (Dual-Data)...")

def _assemble(df_in, y_in):
    # Use deterministic MASTER_COLS
    id_col = df_in[["ID"]].reset_index(drop=True)
    df = df_in[MASTER_COLS].copy().reset_index(drop=True)
    return df, id_col, y_in.reset_index(drop=True)

X_tv_master, tv_ids, y_tv_clean = _assemble(X_tv_alois, y_tv)
X_te_master, te_ids, y_te_clean = _assemble(X_te_alois, y_te)


# =============================================================================
# 1. TREE VIEW — Raw, unscaled values
# =============================================================================
train_tree = pd.concat([tv_ids, X_tv_master.copy(), y_tv_clean], axis=1)
test_tree  = pd.concat([te_ids, X_te_master.copy(), y_te_clean], axis=1)

if sota_folds is not None:
    # Bug M2 Fix: Secure merge on ID
    train_tree = train_tree.merge(sota_folds, on="ID", how="left")
    # Clean up position (move fold to the front) - Local cols_tree
    cols_tree = [FOLD_COL] + [c for c in train_tree.columns if c != FOLD_COL]
    train_tree = train_tree[cols_tree]

# --- Save Master View (Unified Bible for Pipeline X) ---
# This file contains ALL 5000 rows, all 15 features, and the fold assignments.
master_path = os.path.join(OUT_DIR, "Master_Needs_SOTA_X.csv")

# Create a final dataframe using the data contract expectation:
# ID | FEATURE_COLS | TARGET_COLS | stratified_fold
# We need to recover the fold column from the original base file
X_all = _alois_features(X_all_raw, train_ref=X_tv_raw)
base_df = pd.read_csv(SOTA_CSV)
master_df = X_all[MASTER_COLS].copy()
master_df.insert(0, "ID", X_all["ID"])
for target in TARGET_COLS:
    master_df[target] = y_all[target].values
master_df = master_df.merge(base_df[['ID', FOLD_COL]], on='ID', how='left')

master_df.to_csv(master_path, index=False)
print(f"✅ Unified Master Bible saved (15 features): {os.path.basename(master_path)}")

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

if sota_folds is not None:
    train_nn = train_nn.merge(sota_folds, on="ID", how="left")
    # Bug 6 Fix: Local cols_nn calculation
    cols_nn = [FOLD_COL] + [c for c in train_nn.columns if c != FOLD_COL]
    train_nn = train_nn[cols_nn]

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
    "  Wealth_Age_Ratio_log  → Log(1 + Wealth / (Age - 17)): wealth accumulation speed during adult years",
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
print("✅ 01x_feature_engineering.py COMPLETE (SYMMETRIC HARDENING)")
print("=" * 70)
