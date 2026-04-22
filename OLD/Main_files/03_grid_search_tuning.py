"""
=============================================================================
STEP 03 - HYPERPARAMETER TUNING (GRID SEARCH)
=============================================================================
PURPOSE:
    Take the top-performing algorithm families identified in Step 02
    (Random Forest, XGBoost, SVM) and systematically search a predefined
    grid of hyperparameters to find better configurations.

    GridSearchCV works by exhaustively testing every combination of
    parameters in the grid, using cross-validation to score each one.
    It is thorough but slow — the search space grows exponentially with
    the number of parameters and values.

WHY ONLY THESE THREE MODELS?
    - Logistic Regression and Naive Bayes have minimal tunable parameters.
    - KNN and Decision Trees showed weaker ROC-AUC in Step 02.
    - RF, XGBoost, SVM showed the best baseline scores and have rich
      hyperparameter spaces worth exploring.

INPUTS:
    - Dataset2_Needs.xls  (Needs sheet, engineered features)
    - utils.py

OUTPUTS:
    - Output/03_grid_search/03_grid_search_results.csv          (comparison table)
    - Output/03_grid_search/03_grid_search_{target}_rf.pkl      (best RF model, serialized)
    - Output/03_grid_search/03_grid_search_{target}_xgb.pkl     (best XGBoost model, serialized)
    - Output/03_grid_search/03_grid_search_{target}_svm.pkl     (best SVM model, serialized)

    To reload a saved model later:
        import joblib
        model = joblib.load("Output/03_grid_search/03_grid_search_AccumulationInvestment_xgb.pkl")
=============================================================================
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tabulate import tabulate

# Data Contract: standardized data loading and model evaluation
from utils import load_and_prepare_data, evaluate_model, display_results

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(script_dir, "..", "Dataset2_Needs.xls"))

if not os.path.exists(FILE_PATH):
    print("Error: Could not find Dataset2_Needs.xls.")
    sys.exit(1)

TARGETS = ["AccumulationInvestment", "IncomeInvestment"]

print("=" * 100)
print("STEP 03: HYPERPARAMETER TUNING (GRID SEARCH CV)")
print("=" * 100)

all_tuning_results = []

# ---------------------------------------------------------------------------
# Per-Target Tuning Loop
# ---------------------------------------------------------------------------
# We run the full tuning sequence for each target independently.
# We always use engineered features here — Step 02 proved they improve AUC.
for target in TARGETS:
    print(f"\n\n{'*'*60}\nTUNING TARGET: {target}\n{'*'*60}")

    X_train, X_test, y_train, y_test = load_and_prepare_data(
        FILE_PATH, target, use_engineered_features=True
    )

    # -----------------------------------------------------------------------
    # 1. Random Forest Tuning
    # -----------------------------------------------------------------------
    # Key parameters to tune:
    #   n_estimators    : more trees = less variance, but slower
    #   max_depth       : limiting depth is the main lever against overfitting
    #   min_samples_leaf: higher values force leaves to represent more data points,
    #                     which discourages memorizing rare patterns
    print("\n[1] GridSearchCV for Random Forest...")
    rf_grid = {
        'n_estimators':    [100, 200],
        'max_depth':       [3, 5, 7],
        'min_samples_leaf':[2, 4, 8]
    }
    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_grid,
        cv=5,               # 5-fold stratified cross-validation
        scoring='roc_auc',  # optimize for ranking quality, not raw accuracy
        n_jobs=-1           # use all available CPU cores in parallel
    )
    rf_search.fit(X_train, y_train)
    # Evaluate the single best estimator found by the grid search
    res_rf = evaluate_model(rf_search.best_estimator_, X_train, X_test, y_train, y_test, cv_folds=5)

    all_tuning_results.append({
        "Algorithm":         "GridSearch",
        "Target":            target,
        "Model":             "Random Forest",
        "Best Parameters":   str(rf_search.best_params_),
        "CV ROC-AUC (±Std)": f"{res_rf['cv_metrics']['roc_auc']['mean']:.3f} (±{res_rf['cv_metrics']['roc_auc']['std']:.3f})",
        "Test ROC-AUC":      f"{res_rf['test_metrics']['roc_auc']:.3f}"
    })

    # -----------------------------------------------------------------------
    # 2. XGBoost Tuning
    # -----------------------------------------------------------------------
    # Key parameters to tune:
    #   learning_rate: step size shrinkage — lower = more conservative boosting,
    #                  requires more trees but generalizes better
    #   max_depth    : controls tree complexity per boosting round
    #   n_estimators : number of sequential boosting rounds
    #   subsample    : fraction of training rows sampled per round (row bagging),
    #                  adds stochasticity which reduces overfitting
    print("\n[2] GridSearchCV for XGBoost...")
    xgb_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth':     [3, 5],
        'n_estimators':  [100, 200],
        'subsample':     [0.8, 1.0]
    }
    xgb_search = GridSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=42),
        param_grid=xgb_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    res_xgb = evaluate_model(xgb_search.best_estimator_, X_train, X_test, y_train, y_test, cv_folds=5)

    all_tuning_results.append({
        "Algorithm":         "GridSearch",
        "Target":            target,
        "Model":             "XGBoost",
        "Best Parameters":   str(xgb_search.best_params_),
        "CV ROC-AUC (±Std)": f"{res_xgb['cv_metrics']['roc_auc']['mean']:.3f} (±{res_xgb['cv_metrics']['roc_auc']['std']:.3f})",
        "Test ROC-AUC":      f"{res_xgb['test_metrics']['roc_auc']:.3f}"
    })

    # -----------------------------------------------------------------------
    # 3. SVM Tuning
    # -----------------------------------------------------------------------
    # Key parameters to tune:
    #   C      : regularization strength — smaller C = wider margin, more errors allowed
    #   kernel : 'linear' draws a straight boundary; 'rbf' draws a curved one
    #   gamma  : controls how far the influence of a single training example reaches
    #            ('scale' is a safe default; 'auto' is another heuristic)
    # NOTE: cv=3 here (not 5) because SVM is O(n²) in memory — 3 folds is
    #       already significantly slower than the tree-based models.
    print("\n[3] GridSearchCV for SVM...")
    svm_grid = {
        'C':      [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma':  ['scale', 'auto']
    }
    svm_search = GridSearchCV(
        SVC(probability=True, random_state=42),  # probability=True needed for ROC-AUC
        param_grid=svm_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )
    svm_search.fit(X_train, y_train)
    res_svm = evaluate_model(svm_search.best_estimator_, X_train, X_test, y_train, y_test, cv_folds=5)

    all_tuning_results.append({
        "Algorithm":         "GridSearch",
        "Target":            target,
        "Model":             "SVM",
        "Best Parameters":   str(svm_search.best_params_),
        "CV ROC-AUC (±Std)": f"{res_svm['cv_metrics']['roc_auc']['mean']:.3f} (±{res_svm['cv_metrics']['roc_auc']['std']:.3f})",
        "Test ROC-AUC":      f"{res_svm['test_metrics']['roc_auc']:.3f}"
    })

    # -----------------------------------------------------------------------
    # Model Persistence
    # -----------------------------------------------------------------------
    # Serialize all three best models to disk so they can be reloaded later
    # without re-running the entire (slow) grid search.
    # File naming convention: 03_grid_search_{target}_{algorithm}.pkl
    out_pth = os.path.normpath(os.path.join(script_dir, "..", "Output", "03_grid_search"))
    os.makedirs(out_pth, exist_ok=True)
    joblib.dump(rf_search.best_estimator_,  os.path.join(out_pth, f"03_grid_search_{target}_rf.pkl"))
    joblib.dump(xgb_search.best_estimator_, os.path.join(out_pth, f"03_grid_search_{target}_xgb.pkl"))
    joblib.dump(svm_search.best_estimator_, os.path.join(out_pth, f"03_grid_search_{target}_svm.pkl"))
    print(f" -> Best models serialized to Output/ for target: {target}")

# ---------------------------------------------------------------------------
# Output: Console + CSV
# ---------------------------------------------------------------------------
df_tuning = pd.DataFrame(all_tuning_results)

output_dir = os.path.normpath(os.path.join(script_dir, "..", "Output", "03_grid_search"))
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "03_grid_search_results.csv")
df_tuning.to_csv(csv_path, index=False)

print("\n" + "=" * 120)
print("STEP 03: GRID SEARCH TUNING MASTER TABLE")
print("=" * 120)
print(tabulate(df_tuning, headers='keys', tablefmt='grid', showindex=False))
print(f"\nPM Report saved to: {csv_path}")
