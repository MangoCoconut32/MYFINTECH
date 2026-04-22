"""
=============================================================================
02z_benchmark_baselines.py — SYSTEMATIC MULTI-STAGE BENCHMARK
=============================================================================
PURPOSE:
    Establish a comprehensive performance comparison between two data states:
      1. RAW STAGE        (7 baseline features)
      2. ENGINEERED STAGE (15+ master features from 01z)

    This script evaluates 9 different algorithm families using 5-fold CV
    and reports Full Metrics: Accuracy, Precision, Recall, F1 Macro, and ROC-AUC.

    The "Engineering Lift" is quantified to justify the transition from
    Raw to Engineered feature sets for production.

OUTPUTS:
    - Output/Pipeline_Z/02z_benchmark_results.csv
    - Screen reports for Raw vs Engineered performance.
=============================================================================
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, accuracy_score, make_scorer
)
from sklearn.model_selection import cross_validate

# Core Framework
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier

# Robust Imports for Optional Libraries
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

# Import Data Contract from Main_z
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import utilsz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = utilsz.RANDOM_STATE
TARGET_COLS  = utilsz.TARGET_COLS
OUT_DIR      = utilsz.PIPELINE_Z_DIR

os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
def get_model_registry():
    registry = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN (k=15)":          KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes":         GaussianNB(),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost":            XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
        "EBM (Champion)":      ExplainableBoostingClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    }
    
    if HAS_LGBM:
        registry["LightGBM"] = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    
    if HAS_CATBOOST:
        registry["CatBoost"] = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, allow_writing_files=False)
        
    if HAS_TABPFN:
        registry["TabPFN"] = TabPFNClassifier(device='cpu') # CPU for stability in benchmark
        
    return registry

# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------
def run_benchmark():
    print("=" * 80)
    print("02z — SYSTEMATIC BENCHMARK: RAW vs ENGINEERED")
    print(f"Targeting: {TARGET_COLS}")
    print(f"Libraries: LGBM={HAS_LGBM}, CatBoost={HAS_CATBOOST}, TabPFN={HAS_TABPFN}")
    print("=" * 80)

    registry = get_model_registry()
    results = []

    for target in TARGET_COLS:
        print(f"\n>>> PROCESSING TARGET: {target}")
        
        for stage_name, stage_key in [("Raw", "base"), ("Engineered", "master")]:
            print(f"    [*] Evaluating Stage: {stage_name.upper()}...")
            
            try:
                # Load Data
                X_tv, y_tv_all = utilsz.get_full_train_val(stage=stage_key)
                X_te, y_te_all = utilsz.get_test_set(stage=stage_key)
                
                # Filter for specific target
                y_tv = y_tv_all[target].values
                y_te = y_te_all[target].values
                
                # Feature names (excluding ID)
                f_cols = [c for c in X_tv.columns if c != "ID"]
                X_tv_f = X_tv[f_cols].values
                X_te_f = X_te[f_cols].values
                
                # Get Custom CV Splits from utilsz
                custom_cv = utilsz.get_cv_splitter(stage=stage_key)
                
                for model_name, model in registry.items():
                    start_t = time.time()
                    
                    # 1. 5-Fold Cross Validation
                    scoring = {
                        'auc': 'roc_auc',
                        'f1':  'f1_macro',
                        'acc': 'accuracy',
                        'pre': 'precision',
                        'rec': 'recall'
                    }
                    
                    cv_res = cross_validate(
                        model, X_tv_f, y_tv, 
                        cv=custom_cv, 
                        scoring=scoring,
                        n_jobs=-1 if model_name != "TabPFN" else 1 # TabPFN doesn't like n_jobs
                    )
                    
                    cv_auc = np.mean(cv_res['test_auc'])
                    cv_f1  = np.mean(cv_res['test_f1'])
                    
                    # 2. Final Fit on Train/Val -> Predict on Test
                    model.fit(X_tv_f, y_tv)
                    y_prob = model.predict_proba(X_te_f)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                    
                    test_auc = roc_auc_score(y_te, y_prob)
                    test_f1  = f1_score(y_te, y_pred, average='macro')
                    test_acc = accuracy_score(y_te, y_pred)
                    test_pre = precision_score(y_te, y_pred, zero_division=0)
                    test_rec = recall_score(y_te, y_pred, zero_division=0)
                    
                    duration = time.time() - start_t
                    
                    results.append({
                        "Target": target,
                        "Stage":  stage_name,
                        "Model":  model_name,
                        "CV_AUC": round(cv_auc, 4),
                        "CV_F1":  round(cv_f1, 4),
                        "Test_AUC": round(test_auc, 4),
                        "Test_F1":  round(test_f1, 4),
                        "Test_Acc": round(test_acc, 4),
                        "Test_Pre": round(test_pre, 4),
                        "Test_Rec": round(test_rec, 4),
                        "Duration": round(duration, 2)
                    })
                    print(f"        - {model_name:<20} | AUC: {test_auc:.4f} | F1: {test_f1:.4f} ({duration:.1f}s)")
                    
            except Exception as e:
                print(f"    ⚠️ Failed to process {stage_name} stage: {e}")

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "02z_benchmark_results.csv"), index=False)
    
    for target in TARGET_COLS:
        target_df = df[df["Target"] == target]
        
        # Table A: Full Raw
        print(f"\n\n{'='*30} {target.upper()}: RAW PERFORMANCE {'='*30}")
        raw_df = target_df[target_df["Stage"] == "Raw"].sort_values("Test_AUC", ascending=False)
        print(tabulate(raw_df.drop(columns=["Target", "Stage"]), headers='keys', tablefmt='grid', showindex=False))
        
        # Table B: Full Engineered
        print(f"\n\n{'='*30} {target.upper()}: ENGINEERED PERFORMANCE {'='*30}")
        eng_df = target_df[target_df["Stage"] == "Engineered"].sort_values("Test_AUC", ascending=False)
        print(tabulate(eng_df.drop(columns=["Target", "Stage"]), headers='keys', tablefmt='grid', showindex=False))
        
        # Table C: Cumulative Lift Analysis (AUC & F1)
        print(f"\n\n{'='*30} {target.upper()}: ENGINEERING LIFT ANALYSIS {'='*30}")
        
        lift_rows = []
        models_found = raw_df["Model"].unique()
        for m in models_found:
            m_raw = raw_df[raw_df["Model"] == m].iloc[0]
            m_eng = eng_df[eng_df["Model"] == m].iloc[0] if m in eng_df["Model"].values else None
            
            if m_eng is not None:
                lift_rows.append({
                    "Model": m,
                    "AUC_Raw": m_raw["Test_AUC"],
                    "AUC_Eng": m_eng["Test_AUC"],
                    "Lift_AUC": round(m_eng["Test_AUC"] - m_raw["Test_AUC"], 4),
                    "F1_Raw": m_raw["Test_F1"],
                    "F1_Eng": m_eng["Test_F1"],
                    "Lift_F1": round(m_eng["Test_F1"] - m_raw["Test_F1"], 4)
                })
        
        lift_df = pd.DataFrame(lift_rows).sort_values("AUC_Eng", ascending=False)
        print(tabulate(lift_df, headers='keys', tablefmt='fancy_grid', showindex=False))

    print(f"\n✅ Consolidated CSV saved to: {os.path.join(OUT_DIR, '02z_benchmark_results.csv')}")

if __name__ == "__main__":
    run_benchmark()
