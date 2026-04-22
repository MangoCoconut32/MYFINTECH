"""
=============================================================================
STEP 04 - BAYESIAN HYPERPARAMETER TUNING (OPTUNA)
=============================================================================
PURPOSE:
    Go beyond the fixed grid of Step 03 by using Bayesian Optimization to
    search the continuous hyperparameter space more intelligently.

HOW OPTUNA WORKS (vs GridSearch):
    - GridSearchCV tests every combination you specify — it is exhaustive
      but blind; it treats all combinations as equally likely to be good.
    - Optuna uses a Tree-structured Parzen Estimator (TPE) algorithm.
      It learns from each trial result and focuses subsequent trials on
      regions of the space that are statistically likely to improve the score.
    - This means 30 "smart" Optuna trials often outperforms hundreds of
      GridSearch combinations, especially for continuous parameters like
      learning_rate where a fine-grained value like 0.0124 beats any coarse grid.

WHY ONLY RF AND XGBOOST HERE?
    - SVM's quadratic complexity makes 30-trial Bayesian search prohibitively
      slow on a dataset of ~4000 rows.
    - The grid search results confirmed RF and XGBoost as the dominant families.

INPUTS:
    - Dataset2_Needs.xls  (Needs sheet, engineered features)
    - utils.py

OUTPUTS:
    - Output/04_optuna/04_optuna_results.csv           (best params and AUC per model/target)
    - Output/04_optuna/04_optuna_{target}_rf.pkl       (best RF, serialized)
    - Output/04_optuna/04_optuna_{target}_xgb.pkl      (best XGBoost, serialized)
=============================================================================
"""

import os
import sys
import joblib
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabulate import tabulate

from utils import load_and_prepare_data, evaluate_model, display_results

# Suppress Optuna's per-trial INFO logs — only warnings and errors will print
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(script_dir, "..", "Dataset2_Needs.xls"))

if not os.path.exists(FILE_PATH):
    print("Error: Could not find Dataset2_Needs.xls.")
    sys.exit(1)

TARGETS = ["AccumulationInvestment", "IncomeInvestment"]
N_TRIALS = 30  # Number of Bayesian optimization trials per model/target combination


# ---------------------------------------------------------------------------
# Optuna Objective Functions
# ---------------------------------------------------------------------------
# Each objective function takes a single `trial` object (managed by Optuna)
# and returns a scalar score. Optuna calls this function N_TRIALS times,
# each time suggesting different hyperparameter values to explore.
#
# We use cross_val_score inside the objective to get a robust estimate
# of generalization quality for each hyperparameter set.

def objective_rf(trial, X, y):
    """
    Objective function for Random Forest.
    Optuna will call this N_TRIALS times, each time proposing a set of
    hyperparameters to test. Returns mean CV ROC-AUC (higher = better).
    """
    params = {
        # suggest_int: Optuna will sample an integer in this range
        'n_estimators':     trial.suggest_int('n_estimators', 100, 300, step=50),
        'max_depth':        trial.suggest_int('max_depth', 3, 9),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    # Use stratified k-fold to maintain class balance in each fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1).mean()


def objective_xgb(trial, X, y):
    """
    Objective function for XGBoost.
    Includes a log-scale search for learning_rate, which is more effective
    than a linear scale because small differences near 0 matter more.
    """
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 300, step=50),
        'max_depth':     trial.suggest_int('max_depth', 2, 6),
        # log=True samples on a logarithmic scale: more trials near 0.001 than near 0.2
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        # subsample < 1.0 introduces row bagging, reducing overfitting
        'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1).mean()


# ---------------------------------------------------------------------------
# Main Tuning Loop
# ---------------------------------------------------------------------------
print("=" * 100)
print("STEP 04: ADVANCED BAYESIAN TUNING (OPTUNA)")
print("=" * 100)

all_optuna_results = []

for target in TARGETS:
    print(f"\n\n{'*'*60}\nOPTUNA TUNING TARGET: {target}\n{'*'*60}")
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        FILE_PATH, target, use_engineered_features=True
    )

    # -------------------------------------------------------------------
    # 1. Random Forest — Bayesian Search
    # -------------------------------------------------------------------
    print(f"\n[1] Running Optuna for Random Forest ({N_TRIALS} Bayesian Trials)...")
    study_rf = optuna.create_study(direction="maximize")  # maximize ROC-AUC
    # The lambda wraps our objective so we can pass extra arguments (X, y)
    study_rf.optimize(
        lambda trial: objective_rf(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=False
    )

    # Reconstruct the best model using the winning hyperparameters
    best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42)
    # Final evaluation on train+test using the Data Contract evaluator
    res_rf = evaluate_model(best_rf, X_train, X_test, y_train, y_test, cv_folds=5)

    all_optuna_results.append({
        "Algorithm":         "Optuna",
        "Target":            target,
        "Model":             "Random Forest",
        "Best Parameters":   str(study_rf.best_params),
        "CV ROC-AUC (±Std)": f"{res_rf['cv_metrics']['roc_auc']['mean']:.3f} (±{res_rf['cv_metrics']['roc_auc']['std']:.3f})",
        "Test ROC-AUC":      f"{res_rf['test_metrics']['roc_auc']:.3f}"
    })

    # -------------------------------------------------------------------
    # 2. XGBoost — Bayesian Search
    # -------------------------------------------------------------------
    print(f"\n[2] Running Optuna for XGBoost ({N_TRIALS} Bayesian Trials)...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=False
    )

    # Optuna's best_params for XGBoost won't include eval_metric (set during search),
    # so we add it back manually here before constructing the final model.
    params_xgb = study_xgb.best_params.copy()
    params_xgb['eval_metric'] = 'logloss'
    params_xgb['random_state'] = 42

    best_xgb = XGBClassifier(**params_xgb)
    res_xgb = evaluate_model(best_xgb, X_train, X_test, y_train, y_test, cv_folds=5)

    all_optuna_results.append({
        "Algorithm":         "Optuna",
        "Target":            target,
        "Model":             "XGBoost",
        "Best Parameters":   str(study_xgb.best_params),
        "CV ROC-AUC (±Std)": f"{res_xgb['cv_metrics']['roc_auc']['mean']:.3f} (±{res_xgb['cv_metrics']['roc_auc']['std']:.3f})",
        "Test ROC-AUC":      f"{res_xgb['test_metrics']['roc_auc']:.3f}"
    })

    # -------------------------------------------------------------------
    # Model Persistence
    # -------------------------------------------------------------------
    # Save both best models so their exact weights can be reloaded later
    # for ensemble construction (Step 05) or API deployment.
    out_pth = os.path.normpath(os.path.join(script_dir, "..", "Output", "04_optuna"))
    os.makedirs(out_pth, exist_ok=True)
    joblib.dump(best_rf,  os.path.join(out_pth, f"04_optuna_{target}_rf.pkl"))
    joblib.dump(best_xgb, os.path.join(out_pth, f"04_optuna_{target}_xgb.pkl"))
    print(f" -> Optuna-tuned models serialized to Output/ for target: {target}")

# ---------------------------------------------------------------------------
# Output: Console + CSV
# ---------------------------------------------------------------------------
df_optuna = pd.DataFrame(all_optuna_results)

output_dir = os.path.normpath(os.path.join(script_dir, "..", "Output", "04_optuna"))
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "04_optuna_results.csv")
df_optuna.to_csv(csv_path, index=False)

print("\n" + "=" * 120)
print("STEP 04: OPTUNA TUNING MASTER TABLE")
print("=" * 120)
print(tabulate(df_optuna, headers='keys', tablefmt='grid', showindex=False))
print(f"\nPM Report saved to: {csv_path}")
