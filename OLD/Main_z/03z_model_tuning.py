"""
=============================================================================
03z_model_tuning.py — MULTI-MODEL HYPERPARAMETER OPTIMIZATION (WARM START)
=============================================================================
PURPOSE:
    Finds the absolute best model for each target by tuning three top-tier
    algorithm families using evidence-based, narrow search spaces.
    
    NEW FEATURES:
    1. WARM START: If best_params_Z.json exists, Optuna seeds the search 
       with the previous best results to accelerate convergence.
    2. TEST EVALUATION: Every model is evaluated on the blind Test set.
    3. COMPARISON TABLE: A final SOTA table compares all models/targets.

    Algorithms: XGBoost (Hist), LightGBM (Constrained), Random Forest.
=============================================================================
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import optuna
import warnings
from tabulate import tabulate

# Disabilitiamo i warning fastidiosi a livello globale
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Double-Stage Pipeline Z Contract
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import utilsz

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
RANDOM_STATE = utilsz.RANDOM_STATE
TARGET_COLS  = utilsz.TARGET_COLS
OUT_DIR      = utilsz.PIPELINE_Z_DIR
OS_CORES     = -1  # Global parallelization

N_TRIALS     = 5  # PER model/target

os.makedirs(OUT_DIR, exist_ok=True)
PARAMS_FILE = os.path.join(OUT_DIR, "best_params_Z.json")

# ---------------------------------------------------------------------------
# Objective Functions (Narrow SOTA Search Space)
# ---------------------------------------------------------------------------

def objective(trial, X_df, y, cv, model_type):
    """Generic objective for Optuna tuning using Pandas DataFrames."""
    
    if model_type == "XGB":
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 150, 400),
            'max_depth':        trial.suggest_int('max_depth', 3, 7),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': OS_CORES,
            'random_state': RANDOM_STATE,
            'eval_metric': 'logloss',
            'tree_method': 'hist'
        }
        model = XGBClassifier(**params)
        
    elif model_type == "LGBM":
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 350),
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'num_leaves':       trial.suggest_int('num_leaves', 15, 63), 
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq':     trial.suggest_int('bagging_freq', 1, 5),
            'min_child_samples':trial.suggest_int('min_child_samples', 20, 100),
            'random_state': RANDOM_STATE,
            'verbosity': -1,
            'n_jobs': OS_CORES
        }
        model = LGBMClassifier(**params)
        
    elif model_type == "RF":
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 400),
            'max_depth':         trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 2, 10),
            'random_state': RANDOM_STATE,
            'n_jobs': OS_CORES
        }
        model = RandomForestClassifier(**params)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    scores = []
    
    # [LOGGING] Print parameters at the start of the trial
    trial_id = trial.number
    print(f"\n      [Trial {trial_id}] Parameters: {params}")
    
    # FIX WARNING: Usiamo .iloc per mantenere il formato DataFrame (Pandas) ed i nomi delle colonne
    for f_idx, (train_idx, val_idx) in enumerate(cv):
        X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_prob)
        scores.append(auc_score)
        
        # [LOGGING] Print per-fold result
        print(f"        -> Fold {f_idx}: AUC = {auc_score:.4f}")
        
    avg_auc = np.mean(scores)
    print(f"      [Trial {trial_id}] Finished. Avg AUC: {avg_auc:.4f}")
    return avg_auc

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def run_tuning():
    print("=" * 80)
    print("03z — MULTI-MODEL TUNING: WARM START & PERFORMANCE COMPARISON")
    print(f"Targeting: {TARGET_COLS}")
    print("=" * 80)

    # 1. Load Data
    X_tv, y_tv_all = utilsz.get_full_train_val(stage="master")
    X_te, y_te_all = utilsz.get_test_set(stage="master")
    f_cols = [c for c in X_tv.columns if c != "ID"]
    
    X_tv_df = X_tv[f_cols]
    X_te_df = X_te[f_cols] # For final comparison
    
    cv_splits = utilsz.get_cv_splitter(stage="master")
    
    # 2. Load Existing Params for Warm Start
    existing_results = {}
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r') as f:
                existing_results = json.load(f)
            print(f"[*] Warm Start: Loaded existing parameters from {os.path.basename(PARAMS_FILE)}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load {PARAMS_FILE} for warm start: {e}")

    comparison_results = []
    best_results = existing_results.copy()

    for target in TARGET_COLS:
        print(f"\n>>> [TUNING] Target: {target}")
        y_tv = y_tv_all[target].values
        y_te = y_te_all[target].values
        
        if target not in best_results:
            best_results[target] = {}
        
        for m_type in ["XGB", "LGBM", "RF"]:
            print(f"    [*] Study: {m_type}...")
            
            study = optuna.create_study(direction="maximize")
            
            # WARM START: Se esistevano già parametri migliori per questo (target, modello), inniettiamoli
            if target in existing_results and m_type in existing_results[target]:
                hint_params = existing_results[target][m_type].get("params", {})
                if hint_params:
                    print(f"        -> Seeding study with previous best params...")
                    study.enqueue_trial(hint_params)
            
            study.optimize(lambda t: objective(t, X_tv_df, y_tv, cv_splits, m_type), n_trials=N_TRIALS)
            
            best_auc_cv = study.best_value
            best_params = study.best_params
            
            # 3. Evaluation on Test Set
            print(f"    [*] Evaluating best {m_type} on blind Test set...")
            if m_type == "XGB":
                final_model = XGBClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=OS_CORES, eval_metric='logloss', tree_method='hist')
            elif m_type == "LGBM":
                final_model = LGBMClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=OS_CORES, verbosity=-1)
            else:
                final_model = RandomForestClassifier(**best_params, random_state=RANDOM_STATE, n_jobs=OS_CORES)
                
            final_model.fit(X_tv_df, y_tv)
            test_probs = final_model.predict_proba(X_te_df)[:, 1]
            test_auc = roc_auc_score(y_te, test_probs)
            
            # Save results into hierarchy
            best_results[target][m_type] = {
                "best_auc_cv": round(best_auc_cv, 5),
                "best_auc_test": round(test_auc, 5),
                "params": best_params
            }
            
            # Filename includes both target and method
            m_filename = f"model_Z_{target.replace('Investment', '')}_{m_type}.joblib"
            joblib.dump(final_model, os.path.join(OUT_DIR, m_filename))
            
            comparison_results.append({
                "Target": target.replace('Investment', ''),
                "Model": m_type,
                "CV AUC": round(best_auc_cv, 4),
                "Test AUC": round(test_auc, 4),
                "Status": "NEW" if m_filename not in os.listdir(OUT_DIR) else "UPDATED"
            })

    # Save Registry
    with open(PARAMS_FILE, 'w') as f:
        json.dump(best_results, f, indent=4)
    
    # 4. FINAL COMPARISON TABLE
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE COMPARISON (TUNED MODELS)")
    print("=" * 80)
    
    report_df = pd.DataFrame(comparison_results)
    
    # Add a marker for the "Champion" per target
    report_df['Is_Champion'] = ""
    for target in report_df['Target'].unique():
        mask = report_df['Target'] == target
        max_idx = report_df[mask]['Test AUC'].idxmax()
        report_df.loc[max_idx, 'Is_Champion'] = "🏆"
    
    print(tabulate(report_df, headers='keys', tablefmt='fancy_grid', showindex=False))
    
    print("\n" + "=" * 80)
    print(f"✅ 03z COMPLETE: Models saved in {OUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    run_tuning()
