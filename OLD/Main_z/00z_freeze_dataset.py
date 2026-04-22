"""
=============================================================================
00z_freeze_dataset.py — DATASET FREEZER FOR PIPELINE Z
=============================================================================
PURPOSE:
    This script creates Dataset_Needs_SOTA.csv — the immutable "Bible" of
    Pipeline Z. It encodes the cross-validation fold assignments directly
    into the CSV, so every downstream script reads the exact same splits
    without recomputing any randomness.

    Once this file exists, NO downstream script ever touches the raw Excel
    again. The frozen CSV is the single source of truth.

SPLIT LOGIC:
    ┌─────────────────────────────────────────┐
    │  Rows 0–3999  →  Train/Val block        │
    │    stratified_fold = 0, 1, 2, 3, 4      │
    │    Stratified on IncomeInvestment        │
    │    (most imbalanced → tightest control)  │
    │                                          │
    │  Rows 4000–4999  →  Blind Test block    │
    │    stratified_fold = -1                  │
    │    NEVER used for fitting                │
    └─────────────────────────────────────────┘

OUTPUT:
    BuisnessCase2/Dataset_Needs_SOTA.csv

RUN ONCE:
    python 00z_freeze_dataset.py
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

RAW_EXCEL  = os.path.join(_PROJECT_ROOT, "Dataset2_Needs.xls")
PIPELINE_Z_DIR = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_Z")
OUTPUT_CSV = os.path.join(PIPELINE_Z_DIR, "Dataset_Needs_SOTA.csv")

# Ensure directory exists
os.makedirs(PIPELINE_Z_DIR, exist_ok=True)

ALL_TARGETS     = ["AccumulationInvestment", "IncomeInvestment"]
TARGET_STRATIFY = "stratify_combined"  # Synthetic target for dual-parity
N_SPLITS        = 5
RANDOM_STATE    = 42
TRAIN_VAL_SIZE  = 4000
TEST_SIZE       = 1000

# ---------------------------------------------------------------------------
# Load raw Excel
# ---------------------------------------------------------------------------
print("=" * 68)
print("00z_freeze_dataset.py — Pipeline Z Dataset Freezer")
print("=" * 68)

if not os.path.exists(RAW_EXCEL):
    print(f"\n❌ ERROR: Raw Excel not found at:\n   {RAW_EXCEL}")
    sys.exit(1)

print(f"\n[1/4] Loading raw Excel: {os.path.basename(RAW_EXCEL)} ...")
df = pd.read_excel(RAW_EXCEL, sheet_name="Needs")
df.columns = df.columns.str.strip()
print(f"      Shape: {df.shape}")

# Validate expected size & ID uniqueness
if len(df) < TRAIN_VAL_SIZE + TEST_SIZE:
    print(f"\n❌ ERROR: Dataset has only {len(df)} rows. Expected ≥ {TRAIN_VAL_SIZE + TEST_SIZE}.")
    sys.exit(1)

if not df["ID"].is_unique:
    print(f"\n❌ ERROR: Client ID column contains duplicates. Integrity compromised.")
    sys.exit(1)
print(f"✅ ID Uniqueness Verified.")

# ---------------------------------------------------------------------------
# Initialize stratified_fold column
# ---------------------------------------------------------------------------
df["stratified_fold"] = -5 # Sentinel value

# Phase 1: Total Stratified Split (80/20)
print(f"      Phase 1: Stratifying Test set (1000 rows) using Dual Targets...")
df["stratify_combined"] = (
    df["AccumulationInvestment"].astype(str) + "_" + 
    df["IncomeInvestment"].astype(str)
)

indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, 
    test_size=TEST_SIZE, 
    stratify=df[TARGET_STRATIFY], 
    random_state=RANDOM_STATE
)

df.iloc[test_idx, df.columns.get_loc("stratified_fold")] = -1

# Phase 2: Stratified 5-Fold on Training block
print(f"      Phase 2: Stratifying 5 folds on training block (4000 rows)...")
df_train = df.iloc[train_idx].copy()
y_stratify_train = df_train[TARGET_STRATIFY].values

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# We must map the relative fold index back to the absolute dataframe index
for fold_id, (_, val_fold_rel_idx) in enumerate(skf.split(np.zeros(len(df_train)), y_stratify_train)):
    # val_fold_rel_idx are indices relative to train_idx
    abs_idx = train_idx[val_fold_rel_idx]
    df.iloc[abs_idx, df.columns.get_loc("stratified_fold")] = fold_id

assert (df["stratified_fold"] == -5).sum() == 0, "❌ Some rows were not assigned a fold!"

# ---------------------------------------------------------------------------
# Validation output
# ---------------------------------------------------------------------------
print("\n[3/4] Validation — class balance per fold:")
print("-" * 75)
print(f"{'Fold':>6}  {'N rows':>7}  {'Acc=1':>8}  {'Inc=1':>8}  {'Both':>6}  {'Inc rate':>10}")
print("-" * 75)

for fold_id in sorted(df["stratified_fold"].unique()):
    mask  = df["stratified_fold"] == fold_id
    block = df[mask]
    n     = len(block)
    acc_pos = block["AccumulationInvestment"].sum()
    inc_pos = block["IncomeInvestment"].sum()
    both_pos = ((block["AccumulationInvestment"] == 1) & (block["IncomeInvestment"] == 1)).sum()
    rate  = inc_pos / n if n > 0 else 0.0
    label = str(fold_id) if fold_id >= 0 else "Test (-1)"
    print(f"{label:>6}  {n:>7}  {int(acc_pos):>8}  {int(inc_pos):>8}  {int(both_pos):>6}  {rate:>10.1%}")

print("-" * 75)

# Smoke test assertions
assert (df["stratified_fold"] == -1).sum() == TEST_SIZE, \
    f"❌ Smoke test FAILED: Test block has {(df['stratified_fold'] == -1).sum()} rows, expected {TEST_SIZE}"

tv_rows = df[df["stratified_fold"] >= 0].shape[0]
assert tv_rows == TRAIN_VAL_SIZE, \
    f"❌ Smoke test FAILED: Train/Val block has {tv_rows} rows, expected {TRAIN_VAL_SIZE}"

# Compare Income rate between Fold 0 and Test Set
fold0_rate = df[df["stratified_fold"] == 0]["IncomeInvestment"].mean()
test_rate  = df[df["stratified_fold"] == -1]["IncomeInvestment"].mean()
delta      = abs(fold0_rate - test_rate)
print(f"\n  Fold 0 Income rate : {fold0_rate:.3f}")
print(f"  Test  Income rate  : {test_rate:.3f}")
print(f"  Δ                  : {delta:.4f}  {'✅ OK' if delta < 0.05 else '⚠️  Large skew — check data order'}")

# Remove synthetic column before saving
df.drop(columns=["stratify_combined"], inplace=True)

# ---------------------------------------------------------------------------
# Save frozen CSV
# ---------------------------------------------------------------------------
print(f"\n[4/4] Saving frozen dataset to:\n   {OUTPUT_CSV} ...")
df.to_csv(OUTPUT_CSV, index=False)
print(f"      Rows : {len(df)}")
print(f"      Cols : {len(df.columns)} (includes 'stratified_fold')")

print("\n" + "=" * 68)
print("✅ Dataset frozen. Dataset_Needs_SOTA.csv is the Pipeline Z Bible.")
print("   Never run this script again unless you deliberately want to")
print("   invalidate all existing model checkpoints.")
print("=" * 68)
