"""
=============================================================================
06x_compliance_audit.py — SOTA GLASSBOX AUDIT ENGINE
=============================================================================
PURPOSE:
    Provides mathematical proof of model superiority and individual 
    accountability for MIFID II / AI Act compliance.

    1. STATISTICAL PROOF: 95% Confidence Intervals via Bootstrap.
    2. INDIVIDUAL EVIDENCE: Local Waterfall plots for audit trails.
    3. JSON SUMMARY: Machine-readable audit certificate.
=============================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.utils import resample
from utilsx import get_test_set, FEATURE_COLS, PIPELINE_X_DIR

# --- 1. Audit Configuration ---
N_ITERATIONS = 1000  # Bootstrap iterations for 95% CI
CLIENTS_TO_AUDIT = [0, 50, 100] # Individual cases for the auditor

# Path setup
MODELS_PATH = {
    "ebm_acc": os.path.join(PIPELINE_X_DIR, "03x_ebm_acc_model.pkl"),
    "ebm_inc": os.path.join(PIPELINE_X_DIR, "03x_ebm_inc_model.pkl"),
    "xgb_acc": os.path.join(PIPELINE_X_DIR, "02x_xgb_acc_calibrated.pkl"),
    "xgb_inc": os.path.join(PIPELINE_X_DIR, "02x_xgb_inc_calibrated.pkl"),
    "tr_acc":  os.path.join(PIPELINE_X_DIR, "03x_ebm_acc_transformer.pkl"),
    "tr_inc":  os.path.join(PIPELINE_X_DIR, "03x_ebm_inc_transformer.pkl")
}

print("=" * 70)
print("06x_compliance_audit.py — SOTA Transparency Audit")
print("=" * 70)

# --- 2. Load Models & Test Set ---
print("\n[1/4] Loading models and blind test set...")
X_test_df, y_test_df = get_test_set(stage="base") #

m = {}
for name, path in MODELS_PATH.items():
    if not os.path.exists(path):
        print(f"❌ Missing: {name} at {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        m[name] = pickle.load(f)

# Hardened features for each architecture
ebm_feats = [c for c in FEATURE_COLS if not c.startswith("Age_bracket")]
xgb_feats = FEATURE_COLS

def get_probs(model, transformer, df, feats):
    X = transformer.transform(df)[feats].values
    return model.predict_proba(X)[:, 1]

# Generating predictions
p_acc_ebm = get_probs(m["ebm_acc"], m["tr_acc"], X_test_df, ebm_feats)
p_inc_ebm = get_probs(m["ebm_inc"], m["tr_inc"], X_test_df, ebm_feats)
p_acc_xgb = get_probs(m["xgb_acc"], m["tr_acc"], X_test_df, xgb_feats)
p_inc_xgb = get_probs(m["xgb_inc"], m["tr_inc"], X_test_df, xgb_feats)

# --- 3. Bootstrap Significance Audit ---
print(f"\n[2/4] Running Bootstrap Audit ({N_ITERATIONS} iterations)...")

def audit_bootstrap(y_true, p_ebm, p_xgb):
    deltas = []
    for i in range(N_ITERATIONS):
        # Resample with replacement to create a "new" test set distribution
        y_b, s_e, s_x = resample(y_true, p_ebm, p_xgb, random_state=i)
        if len(np.unique(y_b)) < 2: continue
        # Calculate Delta AUC
        deltas.append(roc_auc_score(y_b, s_e) - roc_auc_score(y_b, s_x))
    
    ci = np.percentile(deltas, [2.5, 97.5]) # 95% Confidence Interval
    return deltas, ci

d_acc, ci_acc = audit_bootstrap(y_test_df["AccumulationInvestment"].values, p_acc_ebm, p_acc_xgb)
d_inc, ci_inc = audit_bootstrap(y_test_df["IncomeInvestment"].values, p_inc_ebm, p_inc_xgb)

print(f"      Accumulation Δ AUC 95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
print(f"      Income       Δ AUC 95% CI: [{ci_inc[0]:.4f}, {ci_inc[1]:.4f}]")

# Save CI Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(d_acc, fill=True, color="#2E86C1", label=f"Accumulation (CI: {ci_acc[0]:.3f} to {ci_acc[1]:.3f})")
sns.kdeplot(d_inc, fill=True, color="#27AE60", label=f"Income (CI: {ci_inc[0]:.3f} to {ci_inc[1]:.3f})")
plt.axvline(0, color="red", linestyle="--", label="Zero Parity")
plt.title("Statistical Superiority: Glassbox vs XGBoost (Bootstrap AUC Delta)", fontweight="bold")
plt.legend()
plt.savefig(os.path.join(PIPELINE_X_DIR, "06x_audit_ci_dominance.png"), dpi=300)
plt.close()

# --- 4. Individual Accountability (Waterfall Evidence) ---
print("\n[3/4] Generating Individual Evidence Cases (Waterfall Plots)...")

def save_local_audit(ebm_model, transformer, df, idx, target_name):
    # Transform specific client row
    client_row = df.iloc[idx:idx+1]
    client_eng = transformer.transform(client_row)[ebm_feats]
    
    # Extract EBM local explanation (Additive contributions)
    explanation = ebm_model.explain_local(client_eng)
    data = explanation.data(0)
    
    names = data['names']
    scores = data['scores']
    
    # Sort and plot top contributors
    sort_idx = np.argsort(np.abs(scores))[::-1][:8]
    plt.figure(figsize=(10, 6))
    colors = ['#E74C3C' if s < 0 else '#2E86C1' for s in np.array(scores)[sort_idx]]
    plt.barh(np.array(names)[sort_idx], np.array(scores)[sort_idx], color=colors)
    plt.axvline(0, color='black', lw=1)
    plt.title(f"MIFID Audit: Client #{idx} Evidence ({target_name})\n"
              f"Final Probability: {ebm_model.predict_proba(client_eng)[0,1]:.2%}", fontweight="bold")
    plt.xlabel("Feature Contribution (Log-Odds)")
    plt.tight_layout()
    plt.savefig(os.path.join(PIPELINE_X_DIR, f"06x_local_audit_{target_name.lower()}_{idx}.png"))
    plt.close()

for idx in CLIENTS_TO_AUDIT:
    save_local_audit(m["ebm_acc"], m["tr_acc"], X_test_df, idx, "Accumulation")
    save_local_audit(m["ebm_inc"], m["tr_inc"], X_test_df, idx, "Income")

# --- 5. Export Summary JSON ---
print("\n[4/4] Exporting Audit Summary...")
summary = {
    "statistical_audit": {
        "accumulation": {"auc_delta_ci": list(ci_acc), "status": "PROVEN" if ci_acc[0] > 0 else "MARGINAL"},
        "income": {"auc_delta_ci": list(ci_inc), "status": "PROVEN" if ci_inc[0] > 0 else "MARGINAL"}
    },
    "audit_cases": CLIENTS_TO_AUDIT
}
with open(os.path.join(PIPELINE_X_DIR, "06x_audit_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 70)
print("✅ SOTA AUDIT COMPLETE")
print("=" * 70)