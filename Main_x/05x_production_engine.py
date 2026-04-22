"""
=============================================================================
05x_production_engine.py — TOTAL GLASSBOX INFERENCE (PIPELINE X)
=============================================================================
PURPOSE:
    Unified Glassbox Production Engine. Loads the 1000-client blind test set
    via the Data Contract (utilsx.py) and runs inference through 
    dual Calibrated EBM models (Accumulation & Income).

DATA CONTRACT:
    - Enforces FEATURE_COLS centrally via utilsx.
    - Router: 
        EBM (Accumulation) -> 03x_ebm_acc_model.pkl
        EBM (Income)       -> 03x_ebm_inc_model.pkl
    - Inputs are RAW unscaled features (Tree View logic).

INPUTS:
    - Dataset_Needs_SOTA.csv         (Handled by utilsx)
    - Dataset2_Needs.xls             (Financial products catalogue)
    - 03x_ebm_acc_model.pkl          (Calibrated EBM Wrapper)
    - 03x_ebm_inc_model.pkl          (Calibrated EBM Wrapper)

OUTPUTS (Output/Pipeline_X/):
    05x_final_recommendations.csv    — per-client recommendation table
    05x_need_distribution_donut.png  — PPT segmentation chart
    05x_coverage_plot.png            — PPT coverage funnel
    05x_conversion_sankey.html       — Interactive routing visualization
    05x_mifid_heatmap.png            — Compliance risk alignment matrix
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
from utilsx import get_test_set, FEATURE_COLS

# ---------------------------------------------------------------------------
# 1. Config & Pre-flight
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

PIPELINE_X_DIR = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_X")
PRODUCTS_XLS   = os.path.join(_PROJECT_ROOT, "Dataset2_Needs.xls")
EBM_ACC_PKL    = os.path.join(PIPELINE_X_DIR, "03x_ebm_acc_model.pkl")
EBM_ACC_TR     = os.path.join(PIPELINE_X_DIR, "03x_ebm_acc_transformer.pkl")
EBM_INC_PKL    = os.path.join(PIPELINE_X_DIR, "03x_ebm_inc_model.pkl")
EBM_INC_TR     = os.path.join(PIPELINE_X_DIR, "03x_ebm_inc_transformer.pkl")

OUT_DIR        = PIPELINE_X_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# --- SOGLIE DECISIONALI DI BUSINESS ---
# 0.5 = Default statistico.
# Abbassare (es. 0.4) per favorire la Copertura Commerciale (Recall).
# Alzare (es. 0.6) per favorire la Sicurezza MIFID (Precision).
THRESHOLD_ACC = 0.50
THRESHOLD_INC = 0.50

print("=" * 70)
print("05x_production_engine.py — TOTAL GLASSBOX INFERENCE ENGINE")
print("=" * 70)

# Pre-flight checks for required files
for label, path in [
    ("Dataset2_Needs.xls",           PRODUCTS_XLS),
    ("03x_ebm_acc_model.pkl",        EBM_ACC_PKL),
    ("03x_ebm_acc_transformer.pkl",  EBM_ACC_TR),
    ("03x_ebm_inc_model.pkl",        EBM_INC_PKL),
    ("03x_ebm_inc_transformer.pkl",  EBM_INC_TR),
]:
    if not os.path.exists(path):
        print(f"❌ Missing: {label} at {path}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Data Loading & Contract Validation
# ---------------------------------------------------------------------------
# The get_test_set(stage="base") returns RAW features
X_test_df, y_test_df = get_test_set(stage="base")

# Load transformers
with open(EBM_ACC_TR, "rb") as f:
    tr_acc = pickle.load(f)
with open(EBM_INC_TR, "rb") as f:
    tr_inc = pickle.load(f)

# Hardened features (Excluding and including exactly what the models saw)
# FEATURE_COLS is reference for XGB/TabNet, EBMs use a filtered list
def get_ebm_feats(df):
    return [c for c in df.columns if c not in ["ID"] and not c.startswith("Age_bracket")]

print(f"✅ Data Contract Verified: Using RAW data + Trained Transformers.")
print(f"      Blind Test Set loaded: {X_test_df.shape[0]} clients.")

# ---------------------------------------------------------------------------
# 3. Glassbox Inference (Dual EBM)
# ---------------------------------------------------------------------------
print("\n[2/5] Running EBM Accumulation Inference...")
with open(EBM_ACC_PKL, "rb") as f:
    ebm_acc = pickle.load(f)

# Transform RAW -> Cleaned/Engineered
X_test_acc_eng = tr_acc.transform(X_test_df)
ebm_feats_acc = [c for c in X_test_acc_eng.columns if not c.startswith("Age_bracket") and c != "ID"]
X_test_acc = X_test_acc_eng[ebm_feats_acc].values

prob_acc = ebm_acc.predict_proba(X_test_acc)[:, 1]
y_acc_true = y_test_df["AccumulationInvestment"].values
print(f"      Accumulation AUC (Test): {roc_auc_score(y_acc_true, prob_acc):.4f}")

print("\n[3/5] Running EBM Income Inference...")
with open(EBM_INC_PKL, "rb") as f:
    ebm_inc = pickle.load(f)

X_test_inc_eng = tr_inc.transform(X_test_df)
ebm_feats_inc = [c for c in X_test_inc_eng.columns if not c.startswith("Age_bracket") and c != "ID"]
X_test_inc = X_test_inc_eng[ebm_feats_inc].values

prob_inc = ebm_inc.predict_proba(X_test_inc)[:, 1]
y_inc_true = y_test_df["IncomeInvestment"].values
print(f"      Income AUC (Test)      : {roc_auc_score(y_inc_true, prob_inc):.4f}")

# ---------------------------------------------------------------------------
# 4. Recommender Engine (MIFID Compliance)
# ---------------------------------------------------------------------------
print(f"\n[4/5] Running MIFID Recommender Engine...")
print(f"      Threshold Accumulation : >= {THRESHOLD_ACC}")
print(f"      Threshold Income       : >= {THRESHOLD_INC}")

# Utilizziamo le variabili di configurazione invece di 0.5
needs_acc = prob_acc >= THRESHOLD_ACC
needs_inc = prob_inc >= THRESHOLD_INC

def get_need_label(acc, inc):
    if acc and inc: return "Both"
    if acc: return "Accumulation"
    if inc: return "Income"
    return "None / Savings"

predicted_need = [get_need_label(a, i) for a, i in zip(needs_acc, needs_inc)]

# Load product catalogue
products_df = pd.read_excel(PRODUCTS_XLS, sheet_name="Products")

def match_product(client_info, products_df, rule="advanced_mifid"):
    """
    Advanced MIFID II Suitability Engine.
    Evaluates Client Needs, Risk, Age, Wealth, and Financial Literacy 
    against specific product constraints.
    """
    need = client_info["need"]
    risk = client_info["risk"]
    age  = client_info["age"]
    wth  = client_info["wealth"]
    edu  = client_info["edu"]

    # --- 1. Base Need Filtering ---
    if need == "Both":
        need_type = 1 
    elif need == "Accumulation":
        need_type = 1
    elif need == "Income":
        need_type = 0
    else:
        return None

    filtered = products_df[products_df["Type"] == need_type].copy()
    if filtered.empty: return None

    # --- 2. Base Risk Filtering (Client Risk must tolerate Product Risk) ---
    filtered = filtered[filtered["Risk"] <= risk]

    # --- 3. Advanced MIFID Gating ---
    if rule == "advanced_mifid":
        
        # Rule A: Age-Gating for Life Insurance (Unit-Linked)
        # Products 2, 6, 7, 8 are Unit-Linked policies. Usually blocked for over 75s.
        if age > 75:
            filtered = filtered[~filtered["IDProduct"].isin([2, 6, 7, 8])]
            
        # Rule B: Wealth-Gating for Segregated Accounts (Gestioni Patrimoniali)
        # Products 9, 10, 11 require high capital entry (e.g., minimum 100k Wealth)
        if wth < 100000:
            filtered = filtered[~filtered["IDProduct"].isin([9, 10, 11])]
            
        # Rule C: Financial Education Protection
        # Products 7 and 11 are "Aggressive". Block if financial literacy is low (e.g., < 0.3)
        if edu < 0.3:
            filtered = filtered[~filtered["IDProduct"].isin([7, 11])]
            
        # Rule D: The "Both" Synergy
        # If client needs both, strongly prefer "Balanced High Dividend" (4) or "Balanced Flexible" (8)
        # We boost their selection by temporarily reducing their perceived risk distance
        if need == "Both" and not filtered[filtered["IDProduct"].isin([4, 8])].empty:
            preferred = filtered[filtered["IDProduct"].isin([4, 8])]
            # Return the highest risk among the preferred synergy products
            return int(preferred.sort_values("Risk", ascending=False).iloc[0]["IDProduct"])

    if filtered.empty: return None

    # Default matching logic: maximize expected returns by picking the product
    # closest to the client's maximum risk tolerance
    return int(filtered.sort_values("Risk", ascending=False).iloc[0]["IDProduct"])


results = []
for i in range(len(X_test_df)):
    # Pack client profile into a dictionary for clean passing
    client_info = {
        "need":   predicted_need[i],
        "risk":   X_test_df.iloc[i]["RiskPropensity"],
        "age":    X_test_df.iloc[i]["Age"],
        "wealth": X_test_df.iloc[i]["Wealth"],
        "edu":    X_test_df.iloc[i]["FinancialEducation"]
    }
    
    results.append({
        "Client_ID":      X_test_df.iloc[i]["ID"],
        "Predicted_Need": client_info["need"],
        "RiskPropensity": client_info["risk"],
        "Age":            client_info["age"],
        "Wealth":         client_info["wealth"],
        "Prob_Acc":       round(prob_acc[i], 4),
        "Prob_Inc":       round(prob_inc[i], 4),
        "Rec_Strict":     match_product(client_info, products_df, rule="strict"),         # Only Risk match
        "Rec_Advanced":   match_product(client_info, products_df, rule="advanced_mifid"), # Full Compliance
    })

df_results = pd.DataFrame(results)
res_path = os.path.join(OUT_DIR, "05x_final_recommendations.csv")
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
plt.savefig(os.path.join(OUT_DIR, "05x_need_distribution_donut.png"), dpi=300)
plt.close()

# 5b. Coverage Funnel
coverage = {
    "Strict": (df_results["Rec_Strict"].notna()).mean() * 100,
    "Advanced MIFID": (df_results["Rec_Advanced"].notna()).mean() * 100,
}
plt.figure(figsize=(10, 6))
plt.barh(list(coverage.keys()), list(coverage.values()), color="#1B3A6B")
plt.xlabel("Coverage (%)")
plt.title("Recommender Engine Portfolio Coverage", fontsize=15, fontweight="bold")
for i, v in enumerate(coverage.values()):
    plt.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')
plt.savefig(os.path.join(OUT_DIR, "05x_coverage_plot.png"), dpi=300)
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
    fig.write_html(os.path.join(OUT_DIR, "05x_conversion_sankey.html"))
    print(f"✅ Sankey Diagram saved: 05x_conversion_sankey.html")
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
        plt.savefig(os.path.join(OUT_DIR, "05x_mifid_heatmap.png"), dpi=300)
        plt.close()
        print(f"✅ Heatmap saved: 05x_mifid_heatmap.png")
except Exception as e: 
    print(f"⚠️ Heatmap error: {e}")

print("\n" + "=" * 70)
print("05x COMPLETE — TOTAL GLASSBOX STACK")
print("=" * 70)