"""
=============================================================================
STEP 07 - EXPLAINABLE AI (XAI) & REGULATORY COMPLIANCE REPORTING
=============================================================================
PURPOSE:
    European financial regulations (MIFID II / IDD) prohibit institutions from
    making automated recommendations without being able to explain WHY.
    This script produces a formal audit trail that satisfies that requirement.

    We use SHAP and LIME — two mathematically distinct explainability frameworks —
    to ensure our explanations are cross-validated rather than relying on a
    single method.

WHY XGBOOST FOR XAI (NOT THE NEURAL NETWORK)?
    SHAP's TreeExplainer computes exact Shapley values for tree-based models
    in near-linear time. For neural networks, SHAP falls back to slower
    sampling-based approximations. For regulatory reporting, we prefer the
    exact, reproducible explanation on the XGBoost model (ROC-AUC 0.867)
    over an approximate one on the neural network.

    IMPORTANT: This script loads the EXACT serialized model produced by Step 04
    (04_bayesian_optuna.py) via joblib. It does NOT retrain the model.
    This guarantees that the XAI explanations describe the real deployment
    model — not a separately trained approximation with potentially different
    internal structure.

THE FOUR AUDIT ARTIFACTS PRODUCED:

  1. Global SHAP Summary Plot (01_Global_SHAP_Summary.png):
     Shows which features push predictions up or down globally across all clients.
     Each dot is one client; color = feature value; x-axis = SHAP impact.

  2. Permutation Importance (02_Global_Permutation_Importance.png):
     A model-agnostic second opinion: for each feature, it measures how much
     ROC-AUC drops when that feature's values are randomly shuffled.
     Independent validation confirms the SHAP ranking is not an artifact.

  3. Local SHAP Waterfall for Client #10 (03_Local_SHAP_Client_10.png):
     Shows exactly why the model gave client #10 their specific prediction.
     Starts from the base rate and adds/subtracts each feature's contribution.
     This is the "audit card" a compliance officer would review.

  4. LIME Local Explanation for Client #10 (04_Local_LIME_Client_10.html):
     LIME fits a local linear model around client #10's neighborhood and
     shows which features the locally linear approximation weighted most.
     Provides a second, independent local explanation to cross-check SHAP.

  5. PDP / ICE Sweep Analysis (05_Sweep_Analysis_PDP_ICE.png):
     PDP (Partial Dependence Plot): average model response as one feature varies.
     ICE (Individual Conditional Expectation): same plot per individual client.
     Together they reveal whether a feature's effect is uniform or heterogeneous.

INPUTS:
    - Dataset2_Needs.xls  (Needs sheet, engineered features)
    - Output/04_optuna/04_optuna_{TARGET}_xgb.pkl   (best XGBoost from Step 04)
    - utils.py

OUTPUTS:
    - Output/07_XAI_Report/01_Global_SHAP_Summary.png
    - Output/07_XAI_Report/02_Global_Permutation_Importance.png
    - Output/07_XAI_Report/03_Local_SHAP_Client_10.png
    - Output/07_XAI_Report/04_Local_LIME_Client_10.html
    - Output/07_XAI_Report/05_Sweep_Analysis_PDP_ICE.png
=============================================================================
"""

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt

import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from xgboost import XGBClassifier

from utils import load_and_prepare_data

# ---------------------------------------------------------------------------
# Path Resolution & Target Selection
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH  = os.path.normpath(os.path.join(script_dir, "..", "Dataset2_Needs.xls"))

if not os.path.exists(FILE_PATH):
    print("Error: Could not find Dataset2_Needs.xls.")
    sys.exit(1)

# We focus the XAI audit on AccumulationInvestment — the target where XGBoost
# achieves the highest absolute AUC (0.867), making explanations most reliable.
TARGET = "AccumulationInvestment"

print("=" * 80)
print("STEP 07: EXPLAINABLE AI (XAI) & COMPLIANCE REPORTING")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Model Training
# ---------------------------------------------------------------------------
# Load data using the standard Data Contract (engineered features, scaled).
print(f"Loading data for target: {TARGET}...")
X_train, X_test, y_train, y_test = load_and_prepare_data(
    FILE_PATH, TARGET, use_engineered_features=True
)

# Load the best XGBoost model that was serialized by Step 04 (Bayesian Optuna).
# This is the correct pipeline approach: we audit the EXACT model that will be
# deployed, not a re-trained approximation with stale hardcoded parameters.
# If Step 04 has not been run yet, a clear error message will guide the user.
print("Loading reference XGBoost model from Step 04 output...")
model_pkl = os.path.normpath(
    os.path.join(script_dir, "..", "Output", "04_optuna",
                 f"04_optuna_{TARGET}_xgb.pkl")
)
if not os.path.exists(model_pkl):
    print(
        f"ERROR: Serialized model not found at:\n  {model_pkl}\n"
        "Please run 04_bayesian_optuna.py first to generate the model file."
    )
    sys.exit(1)

best_xgb = joblib.load(model_pkl)
print(f" -> Loaded: {os.path.basename(model_pkl)}")

# Output directory for all XAI artifacts — prefixed with 07_ to maintain pipeline order
output_dir = os.path.normpath(os.path.join(script_dir, "..", "Output", "07_XAI_Report"))
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# ---------------------------------------------------------------------------
# 2. Global Explanations (Portfolio-Level Audit)
# ---------------------------------------------------------------------------
print("\n[1] Generating Global Explanations...")

# --- 2A. Global SHAP Summary Plot ---
# TreeExplainer computes exact Shapley values using the tree structure.
# Shapley values are from cooperative game theory: each feature's value is
# its average marginal contribution to the prediction across all possible
# orderings of features. They satisfy desirable axioms (efficiency, symmetry,
# dummy, additivity) that other importance metrics do not guarantee.
print(" -> Computing SHAP values (TreeExplainer)...")
explainer   = shap.TreeExplainer(best_xgb)
shap_values = explainer(X_test)  # returns an Explanation object with .values, .base_values, .data

plt.figure(figsize=(10, 6))
# summary_plot: each row = one feature; dots show client-level SHAP values.
# Color = actual feature value (red = high, blue = low).
# x-axis position = impact on model output (positive = pushes toward predicting 1).
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Global SHAP Summary (Feature Directional Impact)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_Global_SHAP_Summary.png"), dpi=300, bbox_inches='tight')
plt.close()

# --- 2B. Permutation Importance ---
# For each feature: randomly shuffle its values across all test clients,
# measure the resulting drop in ROC-AUC, then restore and repeat n_repeats times.
# The mean drop = the feature's importance. This is model-agnostic and
# serves as an independent cross-check of the SHAP ranking above.
print(" -> Computing Permutation Importance (10 repeats)...")
perm_importance = permutation_importance(
    best_xgb, X_test, y_test,
    scoring='roc_auc',
    n_repeats=10,   # repeat each permutation 10 times to reduce variance
    random_state=42,
    n_jobs=-1
)
# Take only the top-10 most important features for readability
sorted_idx = perm_importance.importances_mean.argsort()[-10:]

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx],
         align='center', color='darkblue')
plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
plt.title("Permutation Importance Validation (Top 10 Features)", fontsize=14)
plt.xlabel("Mean ROC-AUC Decrease when Feature is Shuffled")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_Global_Permutation_Importance.png"), dpi=300, bbox_inches='tight')
plt.close()

# Save the top 2 features for use in the PDP/ICE sweep below
top_2_features = np.array(X_test.columns)[sorted_idx][-2:]

# ---------------------------------------------------------------------------
# 3. Local Explanations (Single-Client Audit)
# ---------------------------------------------------------------------------
# Regulators may request an explanation for a specific client decision.
# Here we use Client #10 as a representative example.
print("\n[2] Generating Local Explanations (Single Client Audit)...")
CLIENT_INDEX = 10
client_data  = X_test.iloc[CLIENT_INDEX]

# --- 3A. Local SHAP Waterfall Plot ---
# The waterfall plot shows the "path" to the prediction for this one client:
#   - Starts at E[f(x)]: the model's average prediction across all training data
#   - Each bar adds or subtracts a feature's SHAP contribution
#   - Ends at f(x): the final predicted probability for this client
print(f" -> SHAP Waterfall for Client #{CLIENT_INDEX}...")
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[CLIENT_INDEX], show=False)
plt.title(f"SHAP Waterfall Vector Analysis (Client #{CLIENT_INDEX})", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"03_Local_SHAP_Client_{CLIENT_INDEX}.png"),
            dpi=300, bbox_inches='tight')
plt.close()

# --- 3B. LIME Tabular Explanation ---
# LIME (Local Interpretable Model-agnostic Explanations) perturbs the client's
# feature values, gets predictions from XGBoost for each perturbation, then fits
# a simple linear model that approximates XGBoost's behavior locally.
# The result is a set of linear coefficients showing which features drove
# the prediction for THIS specific client, independent of SHAP's game theory.
print(f" -> LIME local linear approximation for Client #{CLIENT_INDEX}...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,           # needed to compute feature statistics
    feature_names=X_train.columns.tolist(),
    class_names=['No Need', 'Needs Accumulation'],
    mode='classification',
    random_state=42
)

lime_exp = lime_explainer.explain_instance(
    data_row=client_data.values,
    predict_fn=best_xgb.predict_proba,  # LIME queries the model with perturbed inputs
    num_features=10                      # show the 10 most impactful features
)
# Save as interactive HTML — compliance officers can open this in any browser
lime_exp.save_to_file(os.path.join(output_dir, f"04_Local_LIME_Client_{CLIENT_INDEX}.html"))

# ---------------------------------------------------------------------------
# 4. Sweep Analysis — PDP and ICE Plots
# ---------------------------------------------------------------------------
# For the top 2 most important features (from Permutation Importance above):
#
#   PDP (Partial Dependence Plot): shows the average model prediction as
#   that feature's value sweeps across its range, holding all other features
#   at their mean. Reveals the global trend (e.g., "as Age increases, probability rises").
#
#   ICE (Individual Conditional Expectation): same as PDP but plotted for
#   each individual client rather than the average. Reveals heterogeneity:
#   does the trend hold for all clients, or only some?
#
#   kind='both' overlays PDP on top of ICE curves.
print(f"\n[3] PDP/ICE Sweep Analysis for top features: {list(top_2_features)}...")

fig, ax = plt.subplots(figsize=(14, 6))
PartialDependenceDisplay.from_estimator(
    estimator=best_xgb,
    X=X_test,
    features=top_2_features,
    kind='both',       # show both ICE individual lines and the PDP mean line
    ax=ax,
    subsample=100,     # limit ICE lines to 100 randomly sampled clients for readability
    random_state=42
)
plt.suptitle("Sweep Analysis: Partial Dependence & ICE Plots", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_Sweep_Analysis_PDP_ICE.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print(f"SUCCESS! Compliance XAI Report saved to: {output_dir}")
print("=" * 80)
