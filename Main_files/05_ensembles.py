"""
=============================================================================
STEP 05 - ENSEMBLE CLASSIFIERS (VOTING & STACKING)
=============================================================================
PURPOSE:
    Test whether combining the Optuna-tuned models from Step 04 into
    ensemble architectures can push accuracy beyond any single model.

THE THREE ENSEMBLE STRATEGIES:

  1. Soft Voting:
     Each model outputs a probability. The final decision is the average
     of all probabilities. This leverages the full confidence score of each
     model and is generally the strongest voting approach.

  2. Hard Voting:
     Each model outputs a class label (0 or 1). The majority label wins.
     Information is lost by converting probabilities to binary votes before
     combining them — this typically hurts ROC-AUC.

  3. Stacking (Meta-Learning):
     Base learners (RF + XGBoost) generate out-of-fold predictions during
     training. A meta-learner (Logistic Regression) then learns how to
     optimally combine those predictions. More powerful in theory, but
     the meta-learner can be constrained by its own linear capacity.

EXPECTED OUTCOME (documented in report.md):
    Stacking performs better than Hard Voting but fails to beat standalone
    Optuna XGBoost. The non-linear boosted models are "anchored" by the
    linear meta-learner. The conclusion: ensembles added complexity without
    adding meaningful accuracy on this dataset.

IMPORTANT — PIPELINE DEPENDENCY:
    This script reads the best hyperparameters directly from the pkl files
    serialized by Step 04 (04_bayesian_optuna.py). You MUST run Step 04
    before running this script. The four required pkl files are:
        Output/04_optuna/04_optuna_AccumulationInvestment_rf.pkl
        Output/04_optuna/04_optuna_AccumulationInvestment_xgb.pkl
        Output/04_optuna/04_optuna_IncomeInvestment_rf.pkl
        Output/04_optuna/04_optuna_IncomeInvestment_xgb.pkl

    NOTE: Voting/Stacking wrappers (sklearn) require UNFITTED base estimators
    that they can clone and re-fit internally during cross-validation. We therefore
    call get_params() on the loaded pkl to extract hyperparameters, then
    construct a fresh unfitted instance — guaranteeing the same configuration
    as Optuna found, without passing a pre-fitted object to sklearn.

INPUTS:
    - Dataset2_Needs.xls  (Needs sheet, engineered features)
    - Output/04_optuna/04_optuna_{target}_rf.pkl   (hyperparameters for RF)
    - Output/04_optuna/04_optuna_{target}_xgb.pkl  (hyperparameters for XGBoost)
    - utils.py

OUTPUTS:
    - Output/05_ensembles/05_ensembles_results.csv
    - Output/05_ensembles/05_ensembles_{target}_voting_soft.pkl
    - Output/05_ensembles/05_ensembles_{target}_voting_hard.pkl
    - Output/05_ensembles/05_ensembles_{target}_stacking.pkl
=============================================================================
"""

import os
import sys
import joblib
import pandas as pd
from tabulate import tabulate

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Data Contract
from utils import load_and_prepare_data, evaluate_model

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
print("STEP 05: ENSEMBLE CLASSIFIERS (VOTING & STACKING)")
print("=" * 100)

# ---------------------------------------------------------------------------
# Load Hyperparameters from Step 04 (Bayesian Optuna)
# ---------------------------------------------------------------------------
# Instead of hardcoding the best parameters, we load the serialized models
# from Step 04 and call get_params() to extract their exact configuration.
# This guarantees the ensembles use the SAME hyperparameters that were
# selected by Bayesian optimization — not a stale copy that may drift over time.
#
# NOTE: The Voting/Stacking wrappers need UNFITTED base estimators to clone
# and re-fit internally during cross-validation. We cannot pass a pre-fitted
# model directly. The correct approach is:
#   1. Load the Step 04 pkl to read its hyperparameters via get_params()
#   2. Construct a fresh, unfitted model object with those exact same params

optuna_dir = os.path.normpath(os.path.join(script_dir, "..", "Output", "04_optuna"))

def load_params_from_pkl(path):
    """Load a serialized model from Step 04 and return its hyperparameters."""
    if not os.path.exists(path):
        print(f"ERROR: Required model not found: {path}")
        print("Please run 04_bayesian_optuna.py before running this script.")
        sys.exit(1)
    return joblib.load(path).get_params()

# Load params for both targets — these will be used to construct fresh instances
acc_rf_params  = load_params_from_pkl(os.path.join(optuna_dir, "04_optuna_AccumulationInvestment_rf.pkl"))
acc_xgb_params = load_params_from_pkl(os.path.join(optuna_dir, "04_optuna_AccumulationInvestment_xgb.pkl"))
inc_rf_params  = load_params_from_pkl(os.path.join(optuna_dir, "04_optuna_IncomeInvestment_rf.pkl"))
inc_xgb_params = load_params_from_pkl(os.path.join(optuna_dir, "04_optuna_IncomeInvestment_xgb.pkl"))
print(" -> Hyperparameters loaded from Step 04 pkl files.")

all_ensemble_results = []

# ---------------------------------------------------------------------------
# Per-Target Ensemble Loop
# ---------------------------------------------------------------------------
for target in TARGETS:
    print(f"\n\n{'*'*60}\nENSEMBLE TARGET: {target}\n{'*'*60}")

    X_train, X_test, y_train, y_test = load_and_prepare_data(
        FILE_PATH, target, use_engineered_features=True
    )

    # Select the correct parameter set for this target's "expert" models.
    # We create fresh, unfitted instances — the ensemble wrappers will fit them.
    if target == "AccumulationInvestment":
        expert_rf  = RandomForestClassifier(**acc_rf_params)
        expert_xgb = XGBClassifier(**acc_xgb_params)
    else:
        expert_rf  = RandomForestClassifier(**inc_rf_params)
        expert_xgb = XGBClassifier(**inc_xgb_params)

    # Logistic Regression acts as a fast, interpretable third voter/meta-learner
    expert_lr = LogisticRegression(max_iter=1000, random_state=42)

    # -----------------------------------------------------------------------
    # 1. Soft Voting Classifier
    # -----------------------------------------------------------------------
    # Combines the probability outputs (predict_proba) of all three models.
    # The final probability for each class = average of individual probabilities.
    # n_jobs=-1 fits the base estimators in parallel.
    print("\n[1] Constructing & Evaluating VotingClassifier (Soft)...")
    voting_estimators = [('rf', expert_rf), ('xgb', expert_xgb), ('lr', expert_lr)]
    voting_soft_clf = VotingClassifier(estimators=voting_estimators, voting='soft', n_jobs=-1)

    res_voting_soft = evaluate_model(voting_soft_clf, X_train, X_test, y_train, y_test, cv_folds=5)

    all_ensemble_results.append({
        "Algorithm":         "Voting (Soft)",
        "Target":            target,
        "CV ROC-AUC (±Std)": f"{res_voting_soft['cv_metrics']['roc_auc']['mean']:.3f} (±{res_voting_soft['cv_metrics']['roc_auc']['std']:.3f})",
        "Test Precision":    f"{res_voting_soft['test_metrics']['precision']:.3f}",
        "Test Recall":       f"{res_voting_soft['test_metrics']['recall']:.3f}",
        "Test F1":           f"{res_voting_soft['test_metrics']['f1']:.3f}",
        "Test ROC-AUC":      f"{res_voting_soft['test_metrics']['roc_auc']:.3f}"
    })

    # -----------------------------------------------------------------------
    # 2. Hard Voting Classifier
    # -----------------------------------------------------------------------
    # Each model votes with a class label (0 or 1); the majority wins.
    # The same estimator list is reused — VotingClassifier fits fresh copies internally.
    # WARNING: Hard voting cannot produce ROC-AUC from probabilities without
    #          re-fitting; scikit-learn handles this by using decision_function fallback.
    print("\n[2] Constructing & Evaluating VotingClassifier (Hard)...")
    voting_hard_clf = VotingClassifier(estimators=voting_estimators, voting='hard', n_jobs=-1)

    res_voting_hard = evaluate_model(voting_hard_clf, X_train, X_test, y_train, y_test, cv_folds=5)

    all_ensemble_results.append({
        "Algorithm":         "Voting (Hard)",
        "Target":            target,
        "CV ROC-AUC (±Std)": f"{res_voting_hard['cv_metrics']['roc_auc']['mean']:.3f} (±{res_voting_hard['cv_metrics']['roc_auc']['std']:.3f})",
        "Test Precision":    f"{res_voting_hard['test_metrics']['precision']:.3f}",
        "Test Recall":       f"{res_voting_hard['test_metrics']['recall']:.3f}",
        "Test F1":           f"{res_voting_hard['test_metrics']['f1']:.3f}",
        "Test ROC-AUC":      f"{res_voting_hard['test_metrics']['roc_auc']:.3f}"
    })

    # -----------------------------------------------------------------------
    # 3. Stacking Classifier
    # -----------------------------------------------------------------------
    # Two-level model:
    #   Level 0 (base learners): RF and XGBoost generate out-of-fold predictions.
    #       Out-of-fold means: for each fold, the models predict on data they
    #       were NOT trained on — this prevents leakage into the meta-learner.
    #   Level 1 (meta-learner): Logistic Regression learns how to best
    #       weight the base learner predictions to produce the final output.
    #
    # We deliberately exclude LR from the base learners to keep them non-linear.
    print("\n[3] Constructing & Evaluating StackingClassifier (Meta: Logistic Regression)...")
    stacking_base = [('rf', expert_rf), ('xgb', expert_xgb)]
    stacking_clf = StackingClassifier(
        estimators=stacking_base,
        final_estimator=LogisticRegression(),  # meta-learner
        n_jobs=-1
    )

    res_stacking = evaluate_model(stacking_clf, X_train, X_test, y_train, y_test, cv_folds=5)

    all_ensemble_results.append({
        "Algorithm":         "Stacking (Meta: LR)",
        "Target":            target,
        "CV ROC-AUC (±Std)": f"{res_stacking['cv_metrics']['roc_auc']['mean']:.3f} (±{res_stacking['cv_metrics']['roc_auc']['std']:.3f})",
        "Test Precision":    f"{res_stacking['test_metrics']['precision']:.3f}",
        "Test Recall":       f"{res_stacking['test_metrics']['recall']:.3f}",
        "Test F1":           f"{res_stacking['test_metrics']['f1']:.3f}",
        "Test ROC-AUC":      f"{res_stacking['test_metrics']['roc_auc']:.3f}"
    })

    # -----------------------------------------------------------------------
    # Model Persistence
    # -----------------------------------------------------------------------
    # After evaluate_model(), the ensemble classifiers are fully fitted.
    # We serialize all three so they can be reloaded without retraining.
    out_pth = os.path.normpath(os.path.join(script_dir, "..", "Output", "05_ensembles"))
    os.makedirs(out_pth, exist_ok=True)
    joblib.dump(voting_soft_clf, os.path.join(out_pth, f"05_ensembles_{target}_voting_soft.pkl"))
    joblib.dump(voting_hard_clf, os.path.join(out_pth, f"05_ensembles_{target}_voting_hard.pkl"))
    joblib.dump(stacking_clf,    os.path.join(out_pth, f"05_ensembles_{target}_stacking.pkl"))
    print(f" -> Ensemble models serialized to Output/ for target: {target}")

# ---------------------------------------------------------------------------
# Output: Console + CSV
# ---------------------------------------------------------------------------
df_ensembles = pd.DataFrame(all_ensemble_results)

output_dir = os.path.normpath(os.path.join(script_dir, "..", "Output", "05_ensembles"))
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "05_ensembles_results.csv")
df_ensembles.to_csv(csv_path, index=False)

print("\n" + "=" * 120)
print("STEP 05: ENSEMBLE MASTER TABLE")
print("=" * 120)
print(tabulate(df_ensembles, headers='keys', tablefmt='grid', showindex=False))
print(f"\nPM Report saved to: {csv_path}")
