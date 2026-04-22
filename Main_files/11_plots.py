"""Build the figures from the CSVs in Output/.

Run this after the model scripts are done and you want refreshed plots.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
})

BLUE   = "#1f77b4"                 
RED    = "#d62728"           
GREY   = "#7f7f7f"
ORANGE = "#ff7f0e"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "Output", "11_plots"))
ROOT = os.path.normpath(os.path.join(HERE, "..", "Output"))
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


print("[1/8] leaderboard")

rows = []
def push(model, target, auc):
    rows.append({"Model": model, "Target": target, "AUC": float(auc)})


for _, r in pd.read_csv(f"{ROOT}/04_optuna/04_optuna_results.csv").iterrows():
    push(f"{r['Model']} (Optuna)", r["Target"], r["Test ROC-AUC"])


for _, r in pd.read_csv(f"{ROOT}/02_baselines/02b_catboost_lgbm_results.csv").iterrows():
    if "reference" in r["Model"]:
        continue
    name = "CatBoost" if "CatBoost" in r["Model"] else "LightGBM"
    push(name, r["Target"], r["test_auc"])

for _, r in pd.read_csv(f"{ROOT}/02_baselines/02c_tabpfn_results.csv").iterrows():
    push("TabPFN (zero-shot)", r["Target"], r["Test_ROC_AUC"])

for _, r in pd.read_csv(f"{ROOT}/06_neural_networks/06_neural_nets_results.csv").iterrows():
    push("MTL-MLP (Keras)", r["Target"], r["Test ROC-AUC"])

for _, r in pd.read_csv(f"{ROOT}/06_neural_networks/06b_tabnet_results.csv").iterrows():
    push("TabNet MTL", r["Target"], r["Test ROC-AUC"])

for _, r in pd.read_csv(f"{ROOT}/05_ensembles/05b_ebm_results.csv").iterrows():
    push("EBM (glassbox)", r["Target"], r["Test_ROC_AUC"])

tt = pd.read_csv(f"{ROOT}/08_recommender/08b_two_tower_auc.csv")
push("Two-Tower NN", "AccumulationInvestment", tt.loc[tt.metric == "AUC_Accumulation", "value"].iloc[0])
push("Two-Tower NN", "IncomeInvestment",       tt.loc[tt.metric == "AUC_Income",       "value"].iloc[0])

board = pd.DataFrame(rows).pivot(index="Model", columns="Target", values="AUC")
board = board.sort_values("AccumulationInvestment", ascending=True)

fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(board) + 1)))
y = np.arange(len(board))
w = 0.4
ax.barh(y - w/2, board["AccumulationInvestment"], height=w, color=BLUE, label="Accumulation")
ax.barh(y + w/2, board["IncomeInvestment"],       height=w, color=RED,  label="Income")
for i, (a, b) in enumerate(zip(board["AccumulationInvestment"], board["IncomeInvestment"])):
    ax.text(a + 0.005, i - w/2, f"{a:.3f}", va="center", fontsize=8)
    ax.text(b + 0.005, i + w/2, f"{b:.3f}", va="center", fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels(board.index)
ax.set_xlabel("Test ROC-AUC")
ax.set_xlim(0.5, 0.99)
ax.set_title("Model leaderboard - test ROC-AUC, both targets")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=2)
fig.tight_layout()
save(fig, "01_leaderboard.png")


print("[2/8] bootstrap CIs")

sig = pd.read_csv(f"{ROOT}/09_statistical_significance/09_significance_results.csv")
labels  = ["Acc - Raw", "Acc - Calib", "Inc - Raw", "Inc - Calib"]
means   = [sig.XGB_AUC[0], sig.Calib_AUC[0], sig.XGB_AUC[1], sig.Calib_AUC[1]]
los     = [sig.XGB_CI_lo[0], sig.Calib_CI_lo[0], sig.XGB_CI_lo[1], sig.Calib_CI_lo[1]]
his     = [sig.XGB_CI_hi[0], sig.Calib_CI_hi[0], sig.XGB_CI_hi[1], sig.Calib_CI_hi[1]]
colors  = [BLUE, BLUE, RED, RED]
hatches = ["", "//", "", "//"]
yerr    = [[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, means, yerr=yerr, color=colors, capsize=8, edgecolor="black")
for b, h in zip(bars, hatches):
    b.set_hatch(h)
for i, m in enumerate(means):
    ax.text(i, m + 0.005, f"{m:.3f}", ha="center", fontsize=9)
ax.set_ylim(0.65, 0.95)
ax.set_ylabel("Test ROC-AUC (95% bootstrap CI, n=2000)")
ax.set_title("XGBoost raw vs calibrated - ΔAUC not significant (p > 0.75)")
fig.tight_layout()
save(fig, "02_bootstrap_ci.png")


print("[3/8] pareto front")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, target, color in [(axes[0], "AccumulationInvestment", BLUE),
                          (axes[1], "IncomeInvestment",       RED)]:
    df = pd.read_csv(f"{ROOT}/04_optuna/04b_pareto_{target}.csv")
    ax.scatter(df.brier, df.auc, s=70, c=color, edgecolor="black", alpha=0.85)
    ax.axvline(0.15, ls="--", color=GREY, label="Brier < 0.15")
    pick = df[df.brier < 0.15].sort_values("auc", ascending=False).head(1)
    if len(pick):
        ax.scatter(pick.brier, pick.auc, s=220, marker="*",
                   color=ORANGE, edgecolor="black", label="selected", zorder=10)
    ax.set_xlabel("Brier (lower is better)")
    ax.set_ylabel("ROC-AUC (higher is better)")
    ax.set_title(target)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("Pareto front - multi-objective Optuna", fontweight="bold", x=0.07, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "03_pareto_front.png")


print("[4/8] fairness")

fair = pd.read_csv(f"{ROOT}/10_fairness_precision_at_k/10_fairness_slices.csv")
age = fair[fair.slice_variable == "Age"]
order = ["Young (18-35)", "Mid (36-55)", "Senior (55+)"]

acc = [age[(age.Target == "AccumulationInvestment") & (age.slice == s)].AUC.iloc[0] for s in order]
inc = [age[(age.Target == "IncomeInvestment")       & (age.slice == s)].AUC.iloc[0] for s in order]
overall_acc = fair[fair.Target == "AccumulationInvestment"].Overall_AUC.iloc[0]
overall_inc = fair[fair.Target == "IncomeInvestment"].Overall_AUC.iloc[0]

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(order))
w = 0.4
ax.bar(x - w/2, acc, w, color=BLUE, label=f"Accumulation (overall {overall_acc:.3f})")
ax.bar(x + w/2, inc, w, color=RED,  label=f"Income (overall {overall_inc:.3f})")
ax.axhline(overall_acc, color=BLUE, ls=":", alpha=0.5)
ax.axhline(overall_inc, color=RED,  ls=":", alpha=0.5)
for i, (a, b) in enumerate(zip(acc, inc)):
    ax.text(i - w/2, a + 0.012, f"{a:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, b + 0.012, f"{b:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(order)
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0.45, 1.0)
ax.set_title("Fairness audit by age - Income model collapses on Young/Mid")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
fig.tight_layout()
save(fig, "04_fairness_age.png")


print("[5/8] recommender P@K")

rb = pd.read_csv(f"{ROOT}/10_fairness_precision_at_k/10_precision_at_k.csv")
tt = pd.read_csv(f"{ROOT}/08_recommender/08b_two_tower_precision_at_k.csv")
ks = [1, 2]
rb_p = [rb[rb.K == k]["Precision@K"].iloc[0] for k in ks]
tt_p = [tt[tt.K == k]["Precision@K"].iloc[0] for k in ks]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(ks))
w = 0.35
ax.bar(x - w/2, rb_p, w, color=GREY,   label="Rule-based (XGB-gated)")
ax.bar(x + w/2, tt_p, w, color=ORANGE, label="Two-Tower NN")
for i, (a, b) in enumerate(zip(rb_p, tt_p)):
    ax.text(i - w/2, a + 0.012, f"{a:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, b + 0.012, f"{b:.3f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in ks])
ax.set_ylabel("Precision@K")
ax.set_ylim(0, 0.78)
ax.set_title("Precision@K - Two-Tower vs rule-based")
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "05_recommender_pak.png")


print("[6/8] two-tower training")

hist = pd.read_csv(f"{ROOT}/08_recommender/08b_two_tower_history.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(hist.epoch, hist.train_loss, "o-", color=GREY)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE loss")
axes[0].set_title("Training loss")
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

axes[1].plot(hist.epoch, hist.test_auc_acc, "o-", color=BLUE, label="Accumulation")
axes[1].plot(hist.epoch, hist.test_auc_inc, "o-", color=RED,  label="Income")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test AUC")
axes[1].set_title("Held-out test AUC")
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1].legend(frameon=False)

fig.suptitle("Two-Tower training dynamics", fontweight="bold", x=0.07, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, "06_two_tower_training.png")


print("[7/8] boruta map")

with open(f"{ROOT}/03_grid_search/03b_boruta_selected_features.json") as f:
    boruta = json.load(f)

feats = sorted(set(boruta["AccumulationInvestment"]["accepted"]
                   + boruta["AccumulationInvestment"]["rejected"]
                   + boruta["AccumulationInvestment"]["tentative"]))

def cell(target, feat):
    if feat in boruta[target]["accepted"]:  return  1
    if feat in boruta[target]["tentative"]: return  0
    return -1

mat = np.array([[cell("AccumulationInvestment", f), cell("IncomeInvestment", f)] for f in feats])

fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.32 * len(feats))))
ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
ax.set_yticks(np.arange(len(feats)))
ax.set_yticklabels(feats, fontsize=9)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Accumulation", "Income"])
ax.grid(False)
for i in range(len(feats)):
    for j in range(2):
        v = mat[i, j]
        sym = "✓" if v == 1 else ("?" if v == 0 else "✗")
        col = "white" if v != 0 else "black"
        ax.text(j, i, sym, ha="center", va="center", color=col, fontsize=11)
ax.set_title("Boruta-SHAP feature acceptance per target")
fig.tight_layout()
save(fig, "07_boruta_features.png")


print("[8/8] DFS top-20")

imp = pd.read_csv(f"{ROOT}/03_grid_search/03c_dfs_feature_importance.csv", index_col=0)
top = imp.head(20).sort_values("importance")

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top.index, top.importance, color=ORANGE)
for i, v in enumerate(top.importance):
    ax.text(v + 5, i, f"{int(v)}", va="center", fontsize=9)
ax.set_xlabel("LightGBM importance")
ax.set_title("DFS - top 20 engineered features (interactions dominate)")
fig.tight_layout()
save(fig, "08_dfs_top20.png")


print(f"\nDone. {len(os.listdir(OUT))} files in {OUT}")
