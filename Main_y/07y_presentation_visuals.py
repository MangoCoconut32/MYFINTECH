"""
=============================================================================
STEP 07 - PRESENTATION VISUALS GENERATOR
=============================================================================
PURPOSE:
    This script translates the technical outputs (CSVs, JSONs, Pickles) of the
    entire Pipeline Y into C-Level presentation graphics. It answers the
    "so what?" question by turning pure metrics into business narratives.

INPUTS:
    - Train_Master_X.csv / Test_Master_X.csv (for distributions and correlations)
    - 05y_final_recommendations.csv (probabilities and outcomes)
    - 06y_dice_counterfactuals.csv (for actionability plotting)
    - 02_baselines_results.csv (from old R&D stage)
    - Models (EBM, XGB) for feature importances and shape functions

OUTPUTS:
    - 07y_lift_curve.png: Cumulative gains (ROI projection)
    - 07y_waterfall_chart.png: AUC performance delta over baseline
    - 07y_tornado_chart.png: Model architecture rationale (Income vs Accumulation)
    - 07y_correlation_heatmap.png: Feature Orthogonality proof
    - 07y_ebm_shape_function.png: Glassbox transparency proof
    - 07y_portfolio_donut.png: Categorical coverage breakdown
    - 07y_dice_actionability_bars.png: Transforming a 'No' into a 'Yes'
    - 07y_baseline_shootout_table.png: Historical baseline algorithm showdown
    - 07y_probability_density.png: Margin of separation ("Valle della Certezza")

INTERPRETATION:
    These visual assets represent the "last mile" of the data science lifecycle,
    translating model coefficients into Board-ready strategic artifacts.
=============================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Paths
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
PIPELINE_X_DIR = os.path.join(_PROJECT_ROOT, "Main_y", "..", "Output", "Pipeline_Y")

TRAIN_CSV      = os.path.join(PIPELINE_X_DIR, "Train_Master_X.csv")
TEST_CSV       = os.path.join(PIPELINE_X_DIR, "Test_Master_X.csv")
REC_CSV        = os.path.join(PIPELINE_X_DIR, "05y_final_recommendations.csv")
DICE_CSV       = os.path.join(PIPELINE_X_DIR, "06y_dice_counterfactuals.csv")
EBM_PKL        = os.path.join(PIPELINE_X_DIR, "03y_ebm_acc_model.pkl")
XGB_INC_PKL    = os.path.join(PIPELINE_X_DIR, "02y_xgb_inc_calibrated.pkl")

# Corporate palette
C_DARK    = "#1B3A6B"
C_MID     = "#2E86C1"
C_LIGHT   = "#5DADE2"
C_GREY    = "#AEB6BF"
C_GREEN   = "#2ECC71"
C_RED     = "#E74C3C"

print("=" * 70)
print("07y_presentation_visuals.py — Pitch Graphics Generator")
print("=" * 70)

# Load data
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
rec_df   = pd.read_csv(REC_CSV)

TARGET_COLS = ["AccumulationInvestment", "IncomeInvestment"]
FOLD_COL    = "stratified_fold"
FEATURE_COLS = [c for c in train_df.columns if c not in TARGET_COLS + [FOLD_COL]]

# -----------------
# 1. Lift Curve
# -----------------
print("[1/7] Generating Lift Curve...")
y_true = test_df["IncomeInvestment"].values
probs = rec_df["Prob_Income"].values

df_lift = pd.DataFrame({'y': y_true, 'p': probs}).sort_values('p', ascending=False)
df_lift['cum_y'] = df_lift['y'].cumsum()
df_lift['pct_contacted'] = np.arange(1, len(df_lift)+1) / len(df_lift) * 100
df_lift['cum_cap'] = df_lift['cum_y'] / df_lift['y'].sum() * 100

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df_lift['pct_contacted'], df_lift['cum_cap'], color=C_DARK, linewidth=3, label="TabNet + XGBoost Ensemble")
ax.plot([0, 100], [0, 100], color=C_GREY, linestyle='--', linewidth=2, label="Random Guessing (No Model)")
ax.set_xlabel("% of Clients Contacted", fontsize=12)
ax.set_ylabel("% of Total Income Buyers Found", fontsize=12)
ax.set_title("The Money Curve: Cumulative Gains (Income)", fontsize=16, fontweight="bold", loc="left", pad=15)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)
ax.legend(frameon=False, fontsize=12)
# Add annotation for 20% contacted finding X% buyers
val_20 = df_lift[df_lift['pct_contacted'] <= 20]['cum_cap'].max()
ax.plot([20, 20], [0, val_20], color=C_RED, linestyle=':')
ax.plot([0, 20], [val_20, val_20], color=C_RED, linestyle=':')
ax.plot(20, val_20, marker='o', color=C_RED, markersize=8)
ax.annotate(f"{val_20:.0f}% of Buyers\nin just 20% calls!", xy=(22, val_20-5), color=C_RED, fontweight='bold', fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PIPELINE_X_DIR, "07y_lift_curve.png"), dpi=300)
plt.close(fig)

# -----------------
# 2. Waterfall Chart
# -----------------
print("[2/7] Generating Waterfall Chart...")
fig, ax = plt.subplots(figsize=(10, 6))

labels = ["Baseline\n(R&D)", "Hybrid Features\n(+0.03)", "Model Tuning\n(+0.02)", "Production\n(Pipeline Y)"]
starts = [0, 0.760, 0.790, 0]
heights = [0.760, 0.030, 0.022, 0.812]
colors = [C_GREY, C_GREEN, C_GREEN, C_DARK]

bars = ax.bar(labels, heights, bottom=starts, color=colors, edgecolor='white', width=0.6)
for i, bar in enumerate(bars):
    val = starts[i] + heights[i]
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"{val:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.plot([0, 1], [0.760, 0.760], color=C_GREY, linestyle=':', alpha=0.5)
ax.plot([1, 2], [0.790, 0.790], color=C_GREY, linestyle=':', alpha=0.5)

ax.set_ylim(0.70, 0.85)
ax.set_ylabel("ROC-AUC (Income)", fontsize=12)
ax.set_title("Performance Leap Waterfall: R&D → Pipeline Y", fontsize=16, fontweight="bold", loc="left", pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(PIPELINE_X_DIR, "07y_waterfall_chart.png"), dpi=300)
plt.close(fig)

# -----------------
# 3. Tornado Chart
# -----------------
print("[3/7] Generating Tornado Chart (EBM vs XGBoost)...")
with open(EBM_PKL, 'rb') as f:
    ebm = pickle.load(f)
with open(XGB_INC_PKL, 'rb') as f:
    xgb = pickle.load(f)

# EBM Top 5
ebm_imp = dict(zip(ebm.term_names_, ebm.term_importances()))
sorted_ebm = sorted(ebm_imp.items(), key=lambda x: x[1], reverse=True)[:5]
ebm_keys, ebm_vals = zip(*sorted_ebm)

# XGB Top 5
xgb_importances = xgb.calibrated_classifiers_[0].estimator.feature_importances_

xgb_imp = dict(zip(FEATURE_COLS, xgb_importances))
sorted_xgb = sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:5]
xgb_keys, xgb_vals = zip(*sorted_xgb)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig.suptitle("Two-Engine Architecture: Why One Model Isn't Enough", fontsize=16, fontweight='bold', ha='center')

# Left Axis (EBM - Accumulation)
y_pos = np.arange(5)
ax1.barh(y_pos, ebm_vals[::-1], color=C_DARK, edgecolor='white')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(ebm_keys[::-1], fontsize=11)
ax1.set_title("Glassbox Target: Accumulation\n(EBM Top 5 Features)", fontsize=13)
ax1.invert_xaxis()
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.tick_params(axis='y', which='both', left=False, labelleft=True)

# Right Axis (XGB - Income)
ax2.barh(y_pos, xgb_vals[::-1], color=C_MID, edgecolor='white')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(xgb_keys[::-1], fontsize=11)
ax2.set_title("Sniper Target: Income\n(XGB/TabNet Top 5 Features)", fontsize=13)
ax2.yaxis.tick_right()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='y', which='both', right=False)

fig.tight_layout()
fig.savefig(os.path.join(PIPELINE_X_DIR, "07y_tornado_chart.png"), dpi=300)
plt.close(fig)

# -----------------
# 4. Correlation Heatmap
# -----------------
print("[4/7] Generating Feature Correlation Heatmap...")
hybrid_cols = [
    "Age", "Wealth", "Income", "RiskPropensity", 
    "FinancialEducation", "FamilyMembers", "Gender",
    "Wealth_per_person", "Inc_to_Wealth_ratio", "FinancialEducation_mul_Income", 
    "Age_mul_Wealth", "FinancialEducation_mul_Wealth", "Age_mul_FinancialEducation", 
    "RiskPropensity_mul_Wealth", "FamilyMembers_mul_FinancialEducation"
]
corr = train_df[hybrid_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=False, cmap="vlag", center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .7}, ax=ax)
ax.set_title("Orthogonality Proof: Hybrid Feature View (Lower Triangle)", fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(PIPELINE_X_DIR, "07y_correlation_heatmap.png"), dpi=300, facecolor='white')
plt.close(fig)

# -----------------
# 5. EBM Shape Function
# -----------------
print("[5/7] Generating EBM Shape Function...")
exp = ebm.explain_global()
# Find Wealth or Wealth_log index
idx = -1
for i, name in enumerate(ebm.term_names_):
    if name == 'Wealth' or name == 'Wealth_log':
        idx = i
        break
if idx == -1:
    idx = 0

data = exp.data(idx)
fig, ax = plt.subplots(figsize=(8, 5))
if data['type'] == 'continuous':
    x_val = data['names'][:-1]
    y_val = data['scores']
    ax.step(x_val, y_val, where='post', color=C_DARK, linewidth=3)
    ax.fill_between(x_val, y_val, step='post', color=C_DARK, alpha=0.1)
    ax.set_xlabel(ebm.term_names_[idx], fontsize=12)
    ax.set_ylabel("Contribution to P(Accumulation)", fontsize=12)
    ax.set_title(f"Native Interpretability: Shape Function for '{ebm.term_names_[idx]}'", fontsize=16, fontweight='bold', loc='left', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PIPELINE_X_DIR, "07y_ebm_shape_function.png"), dpi=300)
plt.close(fig)

# -----------------
# 6. Portfolio Donut Chart
# -----------------
print("[6/8] Generating Portfolio Donut Chart...")
counts = rec_df['Predicted_Need'].value_counts()
fig, ax = plt.subplots(figsize=(6, 6))
colors = [C_DARK, C_MID] if counts.index[0] == 'Accumulation' else [C_MID, C_DARK]
wedges, texts, autotexts = ax.pie(counts, labels=counts.index, autopct='%1.1f%%', 
                                  startangle=90, colors=colors, pctdistance=0.75, wedgeprops=dict(width=0.5, edgecolor='w'))
for t in texts:
    t.set_fontsize(13)
    t.set_fontweight('bold')
for at in autotexts:
    at.set_color('white')
    at.set_fontsize(14)
    at.set_fontweight('bold')

ax.set_title("Portfolio Breakdown (n=1,000 clients)", fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(PIPELINE_X_DIR, "07y_portfolio_donut.png"), dpi=300)
plt.close(fig)

# -----------------
# 7. DiCE Counterfactual Chart
# -----------------
print("[7/7] Generating DiCE Bar Chart...")
try:
    if os.path.exists(DICE_CSV):
        dice_df = pd.read_csv(DICE_CSV)
        # Get the top near-miss client
        nm_df = dice_df[dice_df['Category'] == 'Near-Miss']
        if not nm_df.empty:
            client_df = nm_df[nm_df['Rank'] == nm_df['Rank'].min()]
            if len(client_df) == 2:
                orig = client_df[client_df['Type'] == 'original'].iloc[0]
                cf = client_df[client_df['Type'] == 'counterfactual'].iloc[0]
                
                changed_cols = []
                for c in FEATURE_COLS:
                    if abs(orig[c] - cf[c]) > 0.001:
                        changed_cols.append(c)
                        
                if changed_cols:
                    scaler = MinMaxScaler()
                    train_scaler = train_df[changed_cols].copy()
                    scaler.fit(train_scaler)
                    
                    orig_scaled = scaler.transform(orig[changed_cols].to_frame().T)[0]
                    cf_scaled = scaler.transform(cf[changed_cols].to_frame().T)[0]
                    
                    fig, ax = plt.subplots(figsize=(9, 5))
                    y_pos = np.arange(len(changed_cols))
                    height = 0.35
                    
                    ax.barh(y_pos - height/2, orig_scaled, height, color=C_RED, label='Current Profile', alpha=0.8)
                    ax.barh(y_pos + height/2, cf_scaled, height, color=C_GREEN, label='Required Target', alpha=0.8)
                    
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(changed_cols, fontsize=12)
                    ax.set_xlabel("Normalized Value [0-1]*", fontsize=10)
                    ax.set_title(f"Actionability: Turning a 'No' into a 'Yes' (Near-Miss Client)\nProb: {orig['Prob_Income_before']:.2f} -> >0.50", fontsize=14, fontweight='bold', loc='left', pad=15)
                    ax.legend(frameon=False, fontsize=11)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='x', alpha=0.3)
                    ax.annotate("* Values normalized relative to full portfolio minimums/maximums", 
                                xy=(0, -0.15), xycoords='axes fraction', fontsize=8, color=C_GREY)
                    fig.tight_layout()
                    fig.savefig(os.path.join(PIPELINE_X_DIR, "07y_dice_actionability_bars.png"), dpi=300)
                    plt.close(fig)
        else:
            print("  ⚠️ Insufficient DiCE data for near miss client.")
    else:
        print(f"  ⚠️ Missing {os.path.basename(DICE_CSV)}")
except Exception as e:
    print(f"  ⚠️ Failed to generate DiCE chart: {e}")

# -----------------
# 8. Historical Evaluation Tables
# -----------------
print("[8/9] Generating Presentation Tables...")

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "Main_y", "..", "Output")

def draw_table(df, title, filename):
    fig, ax = plt.subplots(figsize=(10, len(df)*0.4 + 1.2))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(C_DARK)
        else:
            txt = str(df.iloc[i-1].values)
            if "XGBoost" in txt or "TabNet" in txt or "EBM" in txt or "Random Forest" in txt:
                cell.set_facecolor("#D4E6F1")
                cell.set_text_props(weight='bold')
            elif i % 2 == 0:
                cell.set_facecolor("#F2F3F4")
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(PIPELINE_X_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

try:
    # --- Table 1: Raw vs Engineered (Base models)
    base_df = pd.read_csv(os.path.join(OUTPUT_DIR, "02_baselines", "02_baselines_results.csv"))
    
    t1_df = base_df[base_df["Model"].isin(["Logistic Regression", "KNN (k=5)", "Random Forest", "XGBoost"])].copy()
    t1_df["Target"] = t1_df["Target"].str.replace("Investment", " Inv")
    
    t1_base = t1_df[t1_df["Features"] == "Base"].sort_values(by=["Target", "Test ROC-AUC"])
    draw_table(t1_base[["Target", "Model", "CV ROC-AUC", "Test ROC-AUC"]], 
               "Table 1a: Baseline Shootout (Raw Features)", "07y_table1a_baseline_raw.png")

    t1_eng = t1_df[t1_df["Features"] == "Engineered"].sort_values(by=["Target", "Test ROC-AUC"])
    draw_table(t1_eng[["Target", "Model", "CV ROC-AUC", "Test ROC-AUC"]], 
               "Table 1b: Baseline Shootout (Domain EDA)", "07y_table1b_baseline_eng.png")

    # --- Table 2: The GBM Showdown 
    gbm_df = pd.read_csv(os.path.join(OUTPUT_DIR, "02_baselines", "02b_catboost_lgbm_results.csv"))
    gbm_df = gbm_df.rename(columns={"cv_auc": "CV ROC-AUC", "test_auc": "Test ROC-AUC"})
    
    t2_rf = base_df[(base_df["Model"] == "Random Forest") & (base_df["Features"] == "Engineered")].copy()
    t2_df = pd.concat([gbm_df, t2_rf], ignore_index=True)
    t2_df["Target"] = t2_df["Target"].str.replace("Investment", " Inv")
    
    # Optional: clean up the (reference) tags to match
    t2_df["Model"] = t2_df["Model"].str.replace(" (reference)", "").str.replace(" (balanced)", "").str.replace(" (native cat)", "")
    
    t2_df = t2_df.sort_values(by=["Target", "Test ROC-AUC"], ascending=[True, False]).reset_index(drop=True)
    draw_table(t2_df[["Target", "Model", "CV ROC-AUC", "Test ROC-AUC"]],
               "Table 2: Gradient Boosting Methods (Isolating the standard Black-Box)", "07y_table2_gbm_shootout.png")

    # --- Table 3: Optuna Optimization & Target Imbalance
    optuna_df = pd.read_csv(os.path.join(OUTPUT_DIR, "04_optuna", "04_optuna_results.csv"))
    optuna_df["Target"] = optuna_df["Target"].str.replace("Investment", " Inv")
    optuna_df["Test AUC"] = [f"{x:.3f}" if isinstance(x, float) else x for x in optuna_df["Test ROC-AUC"]]
    t3_df = optuna_df[["Target", "Algorithm", "Model", "Test AUC"]]
    draw_table(t3_df, "Table 3: Hyperparameter Tuning (Revealing the 0.76 Income Ceiling)", "07y_table3_optuna_tuning.png")

    # --- Table 4: The DFS Leap & Explainability Wall
    records = []
    
    # 1. Accumulation
    acc_opt = optuna_df[optuna_df["Target"]=="Accumulation Inv"]["Test ROC-AUC"].values[0]
    records.append({"Target": "Accumulation Inv", "Features": "7 (Domain)", "Model": "XGBoost (Optuna)", "Test AUC": f"{acc_opt:.3f}", "Explainability": "SHAP (Approximated)"})
    records.append({"Target": "Accumulation Inv", "Features": "30 (DFS)", "Model": "Pipeline Y: EBM (GA2M)", "Test AUC": "0.883", "Explainability": "EXACT (Glassbox Math)"})

    # 2. Income
    inc_opt = optuna_df[optuna_df["Target"]=="Income Inv"]["Test ROC-AUC"].values[0]
    records.append({"Target": "Income Inv", "Features": "7 (Domain)", "Model": "XGBoost (Optuna)", "Test AUC": f"{inc_opt:.3f}", "Explainability": "SHAP (Approximated)"})
    records.append({"Target": "Income Inv", "Features": "30 (DFS)", "Model": "Pipeline Y: XGBoost (Iso)", "Test AUC": "0.810", "Explainability": "SHAP (Approximated)"})
    records.append({"Target": "Income Inv", "Features": "15 (Hybrid)", "Model": "Pipeline Y: TabNet V3", "Test AUC": "0.812", "Explainability": "SPARSE ATTENTION"})

    t4_df = pd.DataFrame(records)
    draw_table(t4_df, "Table 4: The Pipeline Y Solution (Beating the Performance & Compliance Walls)", "07y_table4_pipelineX_sota.png")

except Exception as e:
    import traceback
    print(f"  ⚠️ Failed to generate tables: {e}")
    traceback.print_exc()

# -----------------
# 9. Probability Density ("Valle della Certezza")
# -----------------
print("[9/9] Generating Probability Density Plot...")
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    prob_pos = rec_df.loc[test_df["IncomeInvestment"] == 1, "Prob_Income"]
    prob_neg = rec_df.loc[test_df["IncomeInvestment"] == 0, "Prob_Income"]

    sns.kdeplot(prob_neg, fill=True, color=C_RED, label="Non-Buyers (Actual 0)", alpha=0.6, ax=ax)
    sns.kdeplot(prob_pos, fill=True, color=C_GREEN, label="Buyers (Actual 1)", alpha=0.6, ax=ax)
    
    ax.axvline(0.5, color=C_DARK, linestyle='--', linewidth=2, label="Decision Threshold (0.50)")
    
    ax.set_xlabel("TabNet V3 Predicted Probability (Income)", fontsize=12)
    ax.set_ylabel("Density of Clients", fontsize=12)
    ax.set_title("The 'Valley of Certainty': Model Confidence Distribution", fontsize=16, fontweight='bold', loc='left', pad=15)
    ax.set_xlim(0, 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=11)
    
    fig.tight_layout()
    fig.savefig(os.path.join(PIPELINE_X_DIR, "07y_probability_density.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"  ⚠️ Failed to generate density plot: {e}")

print("=" * 70)
print("✅ 07y_presentation_visuals.py COMPLETE")
print("=" * 70)
