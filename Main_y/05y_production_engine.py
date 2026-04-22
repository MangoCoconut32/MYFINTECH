"""
=============================================================================
05y_production_engine.py — TOTAL GLASSBOX INFERENCE (PIPELINE X)
=============================================================================
PURPOSE:
    Unified Glassbox Production Engine. Loads the 1000-client blind test set
    via the Data Contract (utilsy.py) and runs inference through 
    dual Calibrated EBM models (Accumulation & Income).

DATA CONTRACT:
    - Enforces FEATURE_COLS centrally via utilsy.
    - Router: 
        EBM (Accumulation) -> 03y_ebm_acc_model.pkl
        EBM (Income)       -> 03y_ebm_inc_model.pkl
    - Inputs are RAW unscaled features (Tree View logic).

INPUTS:
    - Dataset_Needs_SOTA.csv         (Handled by utilsy)
    - Dataset2_Needs.xls             (Financial products catalogue)
    - 03y_ebm_acc_model.pkl          (Calibrated EBM Wrapper)
    - 03y_ebm_inc_model.pkl          (Calibrated EBM Wrapper)

OUTPUTS (Output/Pipeline_Y/):
    05y_final_recommendations.csv    — per-client recommendation table
    05y_need_distribution_donut.png  — PPT segmentation chart
    05y_coverage_plot.png            — PPT coverage funnel
    05y_conversion_sankey.html       — Interactive routing visualization
    05y_mifid_heatmap.png            — Compliance risk alignment matrix
=============================================================================
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

# Import the strict Data Contract
from utilsy import get_test_set, FEATURE_COLS

# ---------------------------------------------------------------------------
# 1. Config & Pre-flight
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

PIPELINE_X_DIR = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_Y")
PRODUCTS_XLS   = os.path.join(_PROJECT_ROOT, "Dataset2_Needs.xls")
EBM_ACC_PKL    = os.path.join(PIPELINE_X_DIR, "03y_ebm_acc_model.pkl")
EBM_INC_PKL    = os.path.join(PIPELINE_X_DIR, "03y_ebm_inc_model.pkl")

OUT_DIR        = PIPELINE_X_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("05y_production_engine.py — TOTAL GLASSBOX INFERENCE ENGINE")
print("=" * 70)

# Pre-flight checks for required files
for label, path in [
    ("Dataset2_Needs.xls",        PRODUCTS_XLS),
    ("03y_ebm_acc_model.pkl",     EBM_ACC_PKL),
    ("03y_ebm_inc_model.pkl",     EBM_INC_PKL),
]:
    if not os.path.exists(path):
        print(f"❌ Missing: {label} at {path}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Data Loading & Contract Validation
# ---------------------------------------------------------------------------
print("\n[1/5] Loading Test Set via Data Contract...")

# The get_test_set() returns raw data exactly as needed by the EBMs
X_test_df, y_test_df = get_test_set()

# Peel ID off to pass clean matrices to the models
X_test = X_test_df[FEATURE_COLS].values
y_acc_true = y_test_df["AccumulationInvestment"].values
y_inc_true = y_test_df["IncomeInvestment"].values

print(f"✅ Data Contract Verified: Using {len(FEATURE_COLS)} centralized features.")
print(f"      Blind Test Set loaded: {X_test.shape[0]} clients.")

# ---------------------------------------------------------------------------
# 3. Glassbox Inference (Dual EBM)
# ---------------------------------------------------------------------------
print("\n[2/5] Running EBM Accumulation Inference...")
with open(EBM_ACC_PKL, "rb") as f:
    ebm_acc = pickle.load(f)
prob_acc = ebm_acc.predict_proba(X_test)[:, 1]
print(f"      Accumulation AUC (Test): {roc_auc_score(y_acc_true, prob_acc):.4f}")

print("\n[3/5] Running EBM Income Inference...")
with open(EBM_INC_PKL, "rb") as f:
    ebm_inc = pickle.load(f)
prob_inc = ebm_inc.predict_proba(X_test)[:, 1]
print(f"      Income AUC (Test)      : {roc_auc_score(y_inc_true, prob_inc):.4f}")

# ---------------------------------------------------------------------------
# 4. Recommender Engine (MIFID Compliance)
# ---------------------------------------------------------------------------
print("\n[4/5] Running MIFID Recommender Engine...")

needs_acc = prob_acc >= 0.5
needs_inc = prob_inc >= 0.5

def get_need_label(acc, inc):
    if acc and inc: return "Both"
    if acc: return "Accumulation"
    if inc: return "Income"
    return "None / Savings"

predicted_need = [get_need_label(a, i) for a, i in zip(needs_acc, needs_inc)]

# Load product catalogue
products_df = pd.read_excel(PRODUCTS_XLS, sheet_name="Products")

def match_product(client_need, client_risk, client_age, products_df, rule="strict"):
    if client_need == "Both":
        need_type = 1 # Prioritize Accumulation if both
    elif client_need == "Accumulation":
        need_type = 1
    elif client_need == "Income":
        need_type = 0
    else:
        return None

    # Age-gating rule (MIFID): Prevent high-risk long-term income products for Seniors
    if rule == "age_gated":
        if client_need == "Income" and client_age > 65:
            client_risk = min(client_risk, 0.4)
        rule = "strict"

    filtered = products_df[products_df["Type"] == need_type].copy()
    if filtered.empty: return None

    if rule == "strict":
        valid = filtered[filtered["Risk"] <= client_risk]
        if valid.empty: return None
        return int(valid.sort_values("Risk", ascending=False).iloc[0]["IDProduct"])
    
    elif rule == "closest":
        filtered["_dist"] = (filtered["Risk"] - client_risk).abs()
        return int(filtered.sort_values("_dist").iloc[0]["IDProduct"])
    
    return None

results = []
for i in range(len(X_test_df)):
    client_id = X_test_df.iloc[i]["ID"]
    need = predicted_need[i]
    risk = X_test_df.iloc[i]["RiskPropensity"]
    age  = X_test_df.iloc[i]["Age"]
    
    results.append({
        "Client_ID": client_id,
        "Predicted_Need": need,
        "RiskPropensity": risk,
        "Age": age,
        "Prob_Acc": round(prob_acc[i], 4),
        "Prob_Inc": round(prob_inc[i], 4),
        "Rec_Strict": match_product(need, risk, age, products_df, "strict"),
        "Rec_AgeGated": match_product(need, risk, age, products_df, "age_gated"),
        "Rec_Closest": match_product(need, risk, age, products_df, "closest"),
    })

df_results = pd.DataFrame(results)
res_path = os.path.join(OUT_DIR, "05y_final_recommendations.csv")
df_results.to_csv(res_path, index=False)
print(f"✅ Final recommendations saved: {os.path.basename(res_path)}")

# ---------------------------------------------------------------------------
# 5. PPT Visualization
# ---------------------------------------------------------------------------
print("\n[5/5] Generating PPT coverage plots...")

# 5a. Donut Chart (Need Distribution)
need_counts = df_results["Predicted_Need"].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(need_counts, labels=need_counts.index, autopct='%1.1f%%', startangle=140, 
        colors=["#1B3A6B", "#2E86C1", "#5DADE2", "#AEB6BF"], wedgeprops=dict(width=0.3))
plt.title("Total Glassbox — Predicted Needs Distribution", fontsize=15, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "05y_need_distribution_donut.png"), dpi=300)
plt.close()

# 5b. Coverage Funnel
coverage = {
    "Strict": (df_results["Rec_Strict"].notna()).mean() * 100,
    "Age-Gated": (df_results["Rec_AgeGated"].notna()).mean() * 100,
    "Closest": (df_results["Rec_Closest"].notna()).mean() * 100
}
plt.figure(figsize=(10, 6))
plt.barh(list(coverage.keys()), list(coverage.values()), color="#1B3A6B")
plt.xlabel("Coverage (%)")
plt.title("Recommender Engine Portfolio Coverage", fontsize=15, fontweight="bold")
for i, v in enumerate(coverage.values()):
    plt.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')
plt.savefig(os.path.join(OUT_DIR, "05y_coverage_plot.png"), dpi=300)
plt.close()

# --- 5c. Sankey Diagram (Plotly Interactive) ---
try:
    import plotly.graph_objects as go
    total = len(df_results)
    needs = df_results["Predicted_Need"].value_counts()
    label_map = {"Accumulation": 1, "Income": 2, "Both": 3, "None / Savings": 4}
    sources, targets, values = [], [], []
    
    for label, count in needs.items():
        sources.append(0); targets.append(label_map[label]); values.append(count)
        if label != "None / Savings":
            mask = df_results["Predicted_Need"] == label
            assigned = df_results[mask]["Rec_Strict"].notna().sum()
            rejected = mask.sum() - assigned
            sources.append(label_map[label]); targets.append(5); values.append(assigned)
            sources.append(label_map[label]); targets.append(6); values.append(rejected)
            
    fig = go.Figure(data=[go.Sankey(
        node = dict(pad = 15, thickness = 20, line = dict(color = "black", width = 0.5),
          label = ["Total Clients", "Accumulation", "Income", "Both", "None", "Product Assigned", "MIFID Rejection"],
          color = ["#AEB6BF", "#1B3A6B", "#2E86C1", "#5DADE2", "#CCD1D1", "#27AE60", "#E74C3C"]),
        link = dict(source = sources, target = targets, value = values))])
    fig.update_layout(title_text="Total Glassbox — Conversion Funnel", font_size=12)
    fig.write_html(os.path.join(OUT_DIR, "05y_conversion_sankey.html"))
    print(f"✅ Sankey Diagram saved: 05y_conversion_sankey.html")
except Exception as e: 
    print(f"⚠️ Sankey error: {e}")

# --- 5d. MIFID Heatmap ---
try:
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    assigned_mask = df_results["Rec_Strict"].notna()
    if assigned_mask.any():
        subset = df_results[assigned_mask].copy()
        subset["Product_Risk"] = subset["Rec_Strict"].map(products_df.set_index("IDProduct")["Risk"])
        heatmap_data = pd.crosstab(subset["Product_Risk"], subset["RiskPropensity"])
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
        plt.title("MIFID Safety Matrix (EBM Engine)", fontsize=14, fontweight="bold")
        plt.savefig(os.path.join(OUT_DIR, "05y_mifid_heatmap.png"), dpi=300)
        plt.close()
        print(f"✅ Heatmap saved: 05y_mifid_heatmap.png")
except Exception as e: 
    print(f"⚠️ Heatmap error: {e}")

print("\n" + "=" * 70)
print("05y COMPLETE — TOTAL GLASSBOX STACK")
print("=" * 70)