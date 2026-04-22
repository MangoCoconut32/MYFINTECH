"""
=============================================================================
STEP 02 - SYSTEMATIC BASELINE EVALUATION
=============================================================================
PURPOSE:
    Establish the performance floor for both prediction targets by running
    every candidate algorithm against two feature variants (raw vs engineered).
    This answers two questions:
      1. Which algorithm families are competitive on this dataset?
      2. Do our EDA-derived engineered features actually help?

INPUTS:
    - Dataset2_Needs.xls  (Needs sheet)
    - utils.py            (data loading, scaling, CV evaluation)

OUTPUTS:
    - Output/02_baselines/02_baselines_results.csv   (all metrics for every combination)

INTERPRETATION:
    Look at "Test ROC-AUC" and "CV ROC-AUC" side by side.
    A large gap between them signals overfitting. The "Engineered" vs "Base"
    delta proves (or disproves) the value of feature engineering.
=============================================================================
"""

import os
import sys
import pandas as pd
from tabulate import tabulate

# Scikit-Learn model classes to benchmark
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Data Contract: single source of truth for data loading and evaluation
from utils import load_and_prepare_data, evaluate_model

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------
# Build an absolute path to the data file relative to this script's location.
# This makes the script portable — it works regardless of the working directory
# from which it is invoked.
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(script_dir, "..", "Dataset2_Needs.xls"))

if not os.path.exists(FILE_PATH):
    print("Error: Could not find Dataset2_Needs.xls.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------
# We evaluate every model against BOTH targets so that Phase 2+ tuning efforts
# are focused on the right algorithms for each prediction objective.
TARGETS = ["AccumulationInvestment", "IncomeInvestment"]

# FEATURE_SETS defines the two conditions of our A/B test:
#   "Base"       -> raw columns from the Excel sheet (no transformations)
#   "Engineered" -> adds log(Wealth), income ratios, age buckets etc. from EDA
FEATURE_SETS = [("Base", False), ("Engineered", True)]

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
# We test a representative spread of algorithm families:
#   - Linear       : Logistic Regression
#   - Distance     : KNN with 4 different k values (to show sensitivity)
#   - Tree         : Decision Tree (depth-capped to prevent trivial memorization)
#   - Ensemble     : Random Forest, XGBoost
#   - Kernel       : SVM (probability=True required for ROC-AUC scoring)
#   - Probabilistic: Gaussian Naive Bayes
models_to_test = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN (k=3)":  KNeighborsClassifier(n_neighbors=3),
    "KNN (k=5)":  KNeighborsClassifier(n_neighbors=5),
    "KNN (k=7)":  KNeighborsClassifier(n_neighbors=7),
    "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
    # max_depth=5 limits leaf explosion and avoids perfect training-set memorization
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    # probability=True enables predict_proba(), which is required for ROC-AUC
    "SVM":         SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    # eval_metric='logloss' suppresses XGBoost's verbose deprecation warning
    "XGBoost":     XGBClassifier(eval_metric='logloss', random_state=42)
}

print("=" * 100)
print("STEP 02: SYSTEMATIC BASELINE EVALUATION")
print("=" * 100)
print("Evaluating across all combinations of Targets and Feature Sets...\n")

all_results = []

# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------
# Outer loops over the 2 targets × 2 feature sets = 4 data slices.
# Inner loop iterates all 10 models, calling evaluate_model() for each.
for target in TARGETS:
    for feature_set_name, use_engineered in FEATURE_SETS:
        print(f"[*] Processing -> Target: {target} | Features: {feature_set_name}")

        # load_and_prepare_data handles the stratified split and scaling.
        # Passing use_engineered_features controls whether EDA features are appended.
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            FILE_PATH, target, use_engineered_features=use_engineered
        )

        for model_name, model_instance in models_to_test.items():

            # evaluate_model: fits the model, runs 5-fold stratified CV,
            # then scores independently on the held-out test set.
            results = evaluate_model(model_instance, X_train, X_test, y_train, y_test, cv_folds=5)

            # Pull the summary statistics we care about from the result dict
            cv_f1_mean  = results['cv_metrics']['f1']['mean']
            cv_f1_std   = results['cv_metrics']['f1']['std']
            cv_roc_mean = results['cv_metrics']['roc_auc']['mean']
            test_f1     = results['test_metrics']['f1']
            test_roc    = results['test_metrics']['roc_auc']

            all_results.append({
                "Target":        target,
                "Features":      feature_set_name,
                "Model":         model_name,
                # Format: "mean (±std)" makes variance immediately visible
                "CV F1 (±Std)":  f"{cv_f1_mean:.3f} (±{cv_f1_std:.3f})",
                "CV ROC-AUC":    f"{cv_roc_mean:.3f}",
                "Test F1":       f"{test_f1:.3f}",
                "Test ROC-AUC":  f"{test_roc:.3f}"
            })

# ---------------------------------------------------------------------------
# Output: Console + CSV
# ---------------------------------------------------------------------------
df_results = pd.DataFrame(all_results)

print("\n" + "=" * 115)
print("STEP 02: GRAND COMPARISON TABLE")
print("=" * 115)
print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))

# Persist results to Output/ for downstream reference and PM reporting
output_dir = os.path.normpath(os.path.join(script_dir, "..", "Output", "02_baselines"))
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "02_baselines_results.csv")
df_results.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")
