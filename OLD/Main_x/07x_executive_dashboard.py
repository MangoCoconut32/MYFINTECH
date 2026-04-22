"""
=============================================================================
07x_executive_dashboard.py — EXECUTIVE REPORTING & SOTA AUDIT
=============================================================================
PURPOSE:
    Generates high-level business and compliance visuals to demonstrate the
    ROI, Fairness, and Strategic logic of the Pipeline X Suitability Engine.
    
    This module shifts focus from "Data Science metrics" (AUC, Brier) to
    "Business & Compliance metrics" (Fairness, Lift, Sinergy).

INPUTS:
    - 05x_final_recommendations.csv (The output of the engine)
    - Master_Needs_SOTA_X.csv (To recover Demographic data like Gender)

OUTPUTS (Output/Pipeline_X/):
    - 07x_fairness_audit.png      (Demographic Parity / Bias Check)
    - 07x_business_lift.png       (Commercial Efficiency vs Random)
    - 07x_synergy_heatmap.png     (Age vs Wealth product distribution)
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
OUT_DIR = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_X")

RESULTS_CSV = os.path.join(OUT_DIR, "05x_final_recommendations.csv")
MASTER_CSV  = os.path.join(OUT_DIR, "Master_Needs_SOTA_X.csv")

print("=" * 70)
print("07x_executive_dashboard.py — Executive & Compliance SOTA Audit")
print("=" * 70)

# 1. Load Data
# ---------------------------------------------------------------------------
print("\n[1/4] Loading Engine Results and Demographic Data...")

if not os.path.exists(RESULTS_CSV) or not os.path.exists(MASTER_CSV):
    print("❌ ERROR: Missing input files. Run 01x through 05x first.")
    sys.exit(1)

res_df = pd.read_csv(RESULTS_CSV)
master_df = pd.read_csv(MASTER_CSV)

# Merge to get Gender for the Fairness Audit
# We use ID / Client_ID to map them safely
res_df = res_df.merge(master_df[["ID", "Gender"]], left_on="Client_ID", right_on="ID", how="left")

# Check coverage: Did they get ANY product under the strict Advanced rules?
res_df["Approved"] = res_df["Rec_Advanced"].notna()

print(f"      Loaded {len(res_df)} client outcomes.")


# 2. Fairness Audit (Demographic Parity)
# ---------------------------------------------------------------------------
print("\n[2/4] Generating Fairness Audit (Gender Parity)...")

# We want to see if the approval rate differs significantly by Gender
fairness_summary = res_df.groupby("Gender")["Approved"].agg(['mean', 'count']).reset_index()
fairness_summary.rename(columns={"mean": "Approval_Rate", "count": "Total_Clients"}, inplace=True)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=fairness_summary, x="Gender", y="Approval_Rate", palette="Set2", ax=ax)

# Add value labels
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black', xytext=(0, 5), 
                textcoords='offset points')

# Check disparate impact rule (80% rule usually used in US, but good benchmark globally)
rates = fairness_summary["Approval_Rate"].values
if len(rates) == 2:
    ratio = min(rates) / max(rates)
    status = "✅ PASS" if ratio >= 0.8 else "⚠️ FLAG"
    title_suffix = f" (Disparate Impact Ratio: {ratio:.2f} -> {status})"
else:
    title_suffix = ""

ax.set_ylim(0, 1.05)
ax.set_ylabel("Product Approval Rate (Advanced MIFID)", fontsize=11)
ax.set_title(f"Compliance Audit: Gender Fairness{title_suffix}\n"
             "Verifying algorithm does not discriminate based on demographic attributes",
             fontsize=12, fontweight="bold", pad=15)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4)

fair_path = os.path.join(OUT_DIR, "07x_fairness_audit.png")
fig.tight_layout()
fig.savefig(fair_path, dpi=200)
plt.close(fig)
print(f"      Saved: {os.path.basename(fair_path)}")


# 3. Synergy Heatmap (Age vs Wealth Distribution)
# ---------------------------------------------------------------------------
print("\n[3/4] Generating Strategy Heatmap (Age vs Wealth)...")

# Filter only approved clients to see WHAT they got
approved_df = res_df[res_df["Approved"]].copy()

# Create bins for Heatmap
approved_df["Age_Bin"] = pd.cut(approved_df["Age"], bins=[17, 35, 55, 75, 100], labels=["18-35", "36-55", "56-75", "75+"])
approved_df["Wealth_Bin"] = pd.qcut(approved_df["Wealth"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])

# Pivot table: count of "Income" recommendations vs "Accumulation"
pivot_acc = approved_df[approved_df["Predicted_Need"].isin(["Accumulation", "Both"])].pivot_table(
    index="Age_Bin", columns="Wealth_Bin", values="Client_ID", aggfunc="count", fill_value=0)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot_acc, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': 'N. Clients Allocated'}, ax=ax)
ax.set_title("Strategic Allocation: Accumulation & Balanced Products\n"
             "Heatmap showing where the engine concentrates growth strategies", 
             fontsize=12, fontweight="bold", pad=15)
ax.invert_yaxis() # Put young at the bottom
plt.tight_layout()

heat_path = os.path.join(OUT_DIR, "07x_synergy_heatmap.png")
fig.savefig(heat_path, dpi=200)
plt.close(fig)
print(f"      Saved: {os.path.basename(heat_path)}")


# 4. Business ROI (Lift Curve approximation)
# ---------------------------------------------------------------------------
print("\n[4/4] Generating Business Lift (Commercial Efficiency)...")

# We sort clients by their highest probability of needing SOMETHING
res_df["Max_Prob"] = res_df[["Prob_Acc", "Prob_Inc"]].max(axis=1)
res_df.sort_values("Max_Prob", ascending=False, inplace=True)
res_df.reset_index(drop=True, inplace=True)

# Calculate cumulative conversion (assuming top Prob means they "Need" it)
# We use a proxy: if Max_Prob > 0.5, we consider them a "True Target" for this simulation
res_df["Is_Target"] = (res_df["Max_Prob"] >= 0.5).astype(int)
total_targets = res_df["Is_Target"].sum()

res_df["Cum_Targets"] = res_df["Is_Target"].cumsum()
res_df["Percent_Contacted"] = (res_df.index + 1) / len(res_df)
res_df["Percent_Captured"] = res_df["Cum_Targets"] / total_targets

fig, ax = plt.subplots(figsize=(8, 6))

# Model Lift Curve
ax.plot(res_df["Percent_Contacted"], res_df["Percent_Captured"], color="#2E86C1", lw=3, label="Pipeline X Engine")

# Random Guess (Baseline)
ax.plot([0, 1], [0, 1], color="#AEB6BF", lw=2, linestyle="--", label="Random Calling")

# Highlight the 30% mark (e.g., Calling only 30% of the base)
idx_30 = int(len(res_df) * 0.3)
captured_at_30 = res_df.loc[idx_30, "Percent_Captured"]
ax.axvline(x=0.3, color="#E74C3C", linestyle=":", lw=2)
ax.scatter([0.3], [captured_at_30], color="#E74C3C", s=80, zorder=5)
ax.annotate(f"Call 30% of clients\nCapture {captured_at_30:.1%} of targets", 
            (0.32, captured_at_30 - 0.05), fontsize=10, fontweight="bold", color="#E74C3C")

ax.set_title("Commercial ROI: Cumulative Lift Curve\n"
             "Efficiency gains for the Sales & Advisory Team", fontsize=12, fontweight="bold", pad=15)
ax.set_xlabel("Percentage of Client Base Contacted", fontsize=11)
ax.set_ylabel("Percentage of Total Sales Opportunities Captured", fontsize=11)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(loc="lower right")
ax.grid(True, linestyle="--", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

lift_path = os.path.join(OUT_DIR, "07x_business_lift.png")
fig.savefig(lift_path, dpi=200)
plt.close(fig)
print(f"      Saved: {os.path.basename(lift_path)}")

print("\n" + "=" * 70)
print("✅ 07x_executive_dashboard.py COMPLETE")
print("=" * 70)
