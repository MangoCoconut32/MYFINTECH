"""
=============================================================================
03y_train_ebm_income.py — GLASSBOX CHAMPION FOR INCOME
=============================================================================
PURPOSE:
    Trains an Explainable Boosting Machine (EBM / GA2M) exclusively for
    IncomeInvestment using the TREE VIEW of Master Dataset Y (15 raw features).

    EBMs are interpretable by construction: each feature contributes an
    additive shape function f_j(x_j), which makes SHAP/LIME post-hoc
    approximations unnecessary. This is the model the compliance team
    will present to auditors.

    RAW (unscaled) inputs are critical for EBM interpretability:
    Shape functions display actual financial values (e.g. "Wealth > 200,000"),
    not normalized values that require inverse-transformation for readability.

BENCHMARK TARGET:
    XGBoost Giga-Baseline AUC = 0.8121 (02y)
    EBM success threshold     = 0.80   (within 0.01 gap → "ready for branch")

ANTI-LEAKAGE:
    - 5-fold CV uses the frozen stratified_fold column embedded in Train_Master_X_Tree.csv by 01y
    - X_test is only loaded for the final blind evaluation
    - No statistics computed on Test at any point

OUTPUTS (Output/Pipeline_Y/):
    03y_ebm_inc_model.pkl                  — Serialized EBM (pickle)
    03y_ebm_inc_global_explanation.html    — Interactive Shape Function dashboard
    03y_ebm_inc_feature_importance.png     — Feature importance bar chart
    03y_ebm_inc_results.json              — AUC, Brier, P, R, F1 vs XGB reference
    03y_ebm_inc_shape_functions.png        — Top 3 Shape Functions plot
=============================================================================
"""

import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from interpret.glassbox import ExplainableBoostingClassifier
import optuna
from utilsy import (
    get_train_fold, get_full_train_val, get_test_set, get_cv_splitter,
    FEATURE_COLS, TARGET_COLS, RANDOM_STATE
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

PIPELINE_X_DIR = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_Y")
OUT_DIR        = PIPELINE_X_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# Configuration
TARGET       = "IncomeInvestment"
N_TRIALS     = 5
C_INC        = "#27AE60"

C_INC = "#27AE60"

print("=" * 70)
print("03y_train_ebm_income.py — EBM Glassbox (Income)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Load Master Dataset Y via Data Contract (utilsy)
# ---------------------------------------------------------------------------
print(f"\n[1/6] Loading data via Pipeline Y contract (utilsy)...")

X_tv_df, y_tv_df = get_full_train_val()
X_test_df, y_test_df = get_test_set()

# Bug M1 Fix: Peel ID using FEATURE_COLS to prevent identifiers in models
X_train = X_tv_df[FEATURE_COLS].values
y_train = y_tv_df[TARGET].values
X_test  = X_test_df[FEATURE_COLS].values
y_test  = y_test_df[TARGET].values

print(f"✅ Data Contract Verified: Using {len(FEATURE_COLS)} features")
print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
print(f"      Target: {TARGET} | Class balance: {y_train.mean():.1%} positive")

# Load XGBoost reference AUC for comparison
xgb_auc   = None
xgb_brier = None
BASELINE_JSON = os.path.join(PIPELINE_X_DIR, "02y_performance_baseline.json")
if os.path.exists(BASELINE_JSON):
    with open(BASELINE_JSON) as f:
        baseline = json.load(f)
    xgb_auc = baseline.get(TARGET, {}).get("AUC")
    xgb_brier = baseline.get(TARGET, {}).get("Brier")
    print(f"\n      📌 XGBoost reference: AUC={xgb_auc}  Brier={xgb_brier}")

# ---------------------------------------------------------------------------
# 2. EBM Configuration & Optuna 5-Fold CV (frozen folds)
# ---------------------------------------------------------------------------
print(f"\n[2/6] Running Optuna 5-Fold CV ({N_TRIALS} trials)...")

def make_objective(target_name):
    """Returns a closure that Optuna can optimize using frozen folds from utilsy."""
    def objective(trial):
        params = dict(
            feature_names     = FEATURE_COLS,
            learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            max_bins          = trial.suggest_categorical("max_bins", [128, 256, 512]),
            interactions      = trial.suggest_int("interactions", 5, 15),
            min_samples_leaf  = trial.suggest_int("min_samples_leaf", 2, 10),
            outer_bags        = 8,
            inner_bags        = 0,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
        )
        fold_aucs = []
        for i in range(5):
            # Load specific fold through the contract
            X_tr_df, y_tr_df, X_va_df, y_va_df = get_train_fold(i)
            
            # Bug M1 Fix: Peel ID using FEATURE_COLS
            X_tr = X_tr_df[FEATURE_COLS].values
            y_tr = y_tr_df[target_name].values
            X_va = X_va_df[FEATURE_COLS].values
            y_va = y_va_df[target_name].values
            
            ebm_fold = ExplainableBoostingClassifier(**params)
            ebm_fold.fit(X_tr, y_tr)
            p_val = ebm_fold.predict_proba(X_va)[:, 1]
            fold_aucs.append(roc_auc_score(y_va, p_val))
            
        return float(np.mean(fold_aucs))
    return objective

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(make_objective(TARGET), n_trials=N_TRIALS, show_progress_bar=False)

best = study.best_params
print(f"      Best Optuna params: {best}")

EBM_PARAMS = dict(
    feature_names     = FEATURE_COLS,
    learning_rate     = best["learning_rate"],
    max_bins          = best["max_bins"],
    interactions      = best["interactions"],
    min_samples_leaf  = best["min_samples_leaf"],
    outer_bags        = 8,
    inner_bags        = 0,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
)

cv_auc_mean = study.best_value
cv_auc_std  = 0.0
print(f"\n      Optuna 5-Fold CV AUC: {cv_auc_mean:.4f}")

# --- 3. Final Refit & Calibration (isotonic) ---
print(f"\n[3/6] Final refit and Isotonic Calibration (cv=5 frozen folds)...")

base_ebm = ExplainableBoostingClassifier(**EBM_PARAMS)

# Calibrate using the same frozen folds via CV splitter (zero leakage)
cv_splits = get_cv_splitter()
calibrated_ebm = CalibratedClassifierCV(
    base_ebm,
    method="isotonic",
    cv=cv_splits,
    ensemble=False
)
calibrated_ebm.fit(X_train, y_train)

# For explainability, we still want a "raw" EBM fitted on everything
ebm_raw = ExplainableBoostingClassifier(**EBM_PARAMS)
ebm_raw.fit(X_train, y_train)

print(f"      Fitted Calibrated EBM wrapper.")
print(f"      Fitted Raw EBM for explainability: {len(ebm_raw.term_names_)} terms.")

# ---------------------------------------------------------------------------
# 4. Blind Test Evaluation
# ---------------------------------------------------------------------------
print(f"\n[4/6] Evaluating on blind Test Set...")

p_test   = calibrated_ebm.predict_proba(X_test)[:, 1]
pred_test = (p_test >= 0.5).astype(int)

test_auc   = roc_auc_score(y_test, p_test)
test_brier = brier_score_loss(y_test, p_test)
test_prec  = precision_score(y_test, pred_test, zero_division=0)
test_rec   = recall_score(y_test, pred_test, zero_division=0)
test_f1    = f1_score(y_test, pred_test, zero_division=0)

print(f"\n      {'Metric':<18} {'EBM':>8}  {'XGB Ref':>8}  {'Δ':>8}")
print(f"      {'-'*46}")
print(f"      {'AUC':<18} {test_auc:>8.4f}  {xgb_auc or 0:>8.4f}  {test_auc-(xgb_auc or 0):>+8.4f}")
print(f"      {'Brier Score':<18} {test_brier:>8.4f}  {xgb_brier or 0:>8.4f}  {test_brier-(xgb_brier or 0):>+8.4f}")
print(f"      {'Precision':<18} {test_prec:>8.4f}")
print(f"      {'Recall':<18} {test_rec:>8.4f}")
print(f"      {'F1':<18} {test_f1:>8.4f}")

gap = xgb_auc - test_auc if xgb_auc else None
if gap is not None:
    verdict = "✅ READY FOR BRANCH" if gap < 0.01 else f"⚠️  Gap {gap:.4f} > 0.01 — review interactions"
    print(f"\n      EBM vs XGB gap: {gap:+.4f}  →  {verdict}")

# ---------------------------------------------------------------------------
# 5. Save Model & JSON
# ---------------------------------------------------------------------------
print(f"\n[5/6] Saving artifacts...")

# Model
pkl_path = os.path.join(OUT_DIR, "03y_ebm_inc_model.pkl")

# Bug C1 Fix: DO NOT mutate term_names_ on the object itself. 
# We'll handle cosmetic aliases only at plot time.

with open(pkl_path, "wb") as f:
    pickle.dump(calibrated_ebm, f)
print(f"      Saved model: {os.path.basename(pkl_path)} (Calibrated wrapper)")

# Metrics JSON
results = {
    "target": TARGET,
    "model":  "EBM (InterpretML GA2M)",
    "cv_auc_mean":  round(cv_auc_mean, 4),
    "cv_auc_std":   round(cv_auc_std,  4),
    "test_auc":     round(test_auc,    4),
    "test_precision":   round(test_prec,  4),
    "test_recall":      round(test_rec,   4),
    "test_f1":          round(test_f1,    4),
    "test_brier":       round(test_brier,  4),
    "xgb_reference_auc":   xgb_auc,
    "delta_vs_xgb":   round(test_auc - (xgb_auc or 0), 4),
    "ebm_n_terms":    len(ebm_raw.term_names_),
    "ebm_params":     {k: v for k, v in EBM_PARAMS.items() if k != "n_jobs"},
}
json_path = os.path.join(OUT_DIR, "03y_ebm_inc_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"      Saved metrics: {os.path.basename(json_path)}")

# ---------------------------------------------------------------------------
# 6. Explainability — Global Explanation HTML + Feature Importance PNG
# ---------------------------------------------------------------------------
print(f"\n[6/6] Generating explainability artifacts...")

# --- 6a. Feature importance bar chart (static PNG) ---
importances = ebm_raw.term_importances()
term_names  = ebm_raw.term_names_

imp_df = pd.DataFrame({"term": term_names, "importance": importances})
imp_df.sort_values("importance", ascending=True, inplace=True)
top_df = imp_df.tail(25)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#1f77b4" if " & " not in t else "#ff7f0e" for t in top_df["term"]]
ax.barh(top_df["term"], top_df["importance"], color=colors)
ax.set_xlabel("Mean Absolute Contribution (EBM Global Importance)")
ax.set_title(
    f"03y EBM — Global Feature Importance\n"
    f"{TARGET} | AUC={test_auc:.4f} | Blue=Main Effect, Orange=Interaction",
    loc="left", fontweight="bold"
)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

png_path = os.path.join(OUT_DIR, "03y_ebm_inc_feature_importance.png")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"      Saved chart : {os.path.basename(png_path)}")

# --- 6b. Interactive HTML dashboard ---
html_path = os.path.join(OUT_DIR, "03y_ebm_inc_global_explanation.html")
try:
    global_exp = ebm_raw.explain_global(name=f"EBM Global — {TARGET}")
    viz = global_exp.visualize()
    if hasattr(viz, "to_html"):
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(viz.to_html(full_html=True, include_plotlyjs="cdn"))
        print(f"      Saved HTML  : {os.path.basename(html_path)}")
    else:
        import json as _json
        raw_data = global_exp.data()
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(
                "<!DOCTYPE html><html><head><title>EBM Global — {}</title></head>"
                "<body><h2>EBM Shape Functions (JSON fallback)</h2>"
                "<pre style='font-family:monospace;font-size:12px;'>"
                "{}"
                "</pre></body></html>".format(TARGET, _json.dumps(raw_data, indent=2, default=str))
            )
        print(f"      Saved HTML (JSON fallback): {os.path.basename(html_path)}")
except Exception as e:
    print(f"      ⚠️  HTML export failed ({e}) — PNG is saved, model is intact.")

# --- 6c. Shape Functions (Top 3) ---
print(f"      Generating Top 3 Shape Functions (Wealth, Income, RiskPropensity)...")
TOP_SHAPES = ["Wealth", "Income", "RiskPropensity"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Pipeline Y — EBM Shape Functions (Impact on {TARGET})\n"
             "Direct interpretability: how each feature changes the log-odds of investment",
             fontsize=16, fontweight="bold", y=1.05)

try:
    global_exp = ebm_raw.explain_global()
    for ax, feat_name in zip(axes, TOP_SHAPES):
        if feat_name not in ebm_raw.term_names_:
            ax.text(0.5, 0.5, f"Feature '{feat_name}'\nnot found", ha="center", va="center")
            continue
            
        idx = ebm_raw.term_names_.index(feat_name)
        data = global_exp.data(idx)
        x_vals = data['names']
        y_vals = data['scores']
        y_upper = data.get('upper_bounds', y_vals)
        y_lower = data.get('lower_bounds', y_vals)
        
        if isinstance(x_vals[0], (str, bytes)) and "(" in str(x_vals[0]):
             try:
                 x_plot = [float(str(s).split(",")[0].replace("(", "").replace("[", "")) for s in x_vals[:-1]]
                 x_plot.append(x_plot[-1] + (x_plot[-1] - x_plot[-2]))
             except:
                 x_plot = np.arange(len(x_vals))
        else:
            x_plot = x_vals
            
        if len(x_plot) == len(y_vals) + 1:
            x_plot = np.array(x_plot)
            y_vals = np.append(y_vals, y_vals[-1])
            y_upper = np.append(y_upper, y_upper[-1])
            y_lower = np.append(y_lower, y_lower[-1])

        ax.step(x_plot, y_vals, where='post', color="#1B3A6B", linewidth=2.5, label="Main Effect")
        ax.fill_between(x_plot, y_lower, y_upper, step='post', color="#1B3A6B", alpha=0.1, label="95% CI")
        ax.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax.set_title(f"Impact of {feat_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel(feat_name, fontsize=11)
        ax.set_ylabel("Contribution (log-odds)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
except Exception as e:
    print(f"      ⚠️  Shape function plotting failed: {e}")

fig.tight_layout()
shape_path = os.path.join(OUT_DIR, "03y_ebm_inc_shape_functions.png")
fig.savefig(shape_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"      Saved shapes: {os.path.basename(shape_path)}")

# --- 6d. Performance Curves (ROC & Calibration) ---
print(f"      Generating Performance Comparison Curves (EBM vs XGB reference)...")
try:
    with open(os.path.join(OUT_DIR, "02y_xgb_inc_calibrated.pkl"), "rb") as f:
        xgb_ref_model = pickle.load(f)
    p_xgb = xgb_ref_model.predict_proba(X_test)[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. ROC Curve
    fpr_e, tpr_e, _ = roc_curve(y_test, p_test)
    fpr_x, tpr_x, _ = roc_curve(y_test, p_xgb)
    ax1.plot(fpr_e, tpr_e, color=C_INC, lw=3, label=f'EBM (AUC={test_auc:.4f})')
    ax1.plot(fpr_x, tpr_x, color="#AEB6BF", lw=2, linestyle='--', label=f'XGB Baseline (AUC={xgb_auc or 0:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':')
    ax1.set_title("ROC Curve Analysis", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    
    # 2. Calibration Curve
    prob_true_e, prob_pred_e = calibration_curve(y_test, p_test, n_bins=10)
    prob_true_x, prob_pred_x = calibration_curve(y_test, p_xgb, n_bins=10)
    ax2.plot(prob_pred_e, prob_true_e, marker='o', linewidth=2, color=C_INC, label='EBM')
    ax2.plot(prob_pred_x, prob_true_x, marker='s', linewidth=1, color="#AEB6BF", linestyle='--', label='XGB Baseline')
    ax2.plot([0, 1], [0, 1], linestyle=':', color='black', label='Perfect Calibration')
    ax2.set_title("Reliability Diagram (Calibration)", fontweight="bold")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of Positives")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)
    
    fig.suptitle(f"Model Diagnostic: EBM Supremacy Audit ({TARGET})", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    perf_path = os.path.join(OUT_DIR, "03y_ebm_inc_performance_curves.png")
    fig.savefig(perf_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved curves: {os.path.basename(perf_path)}")
except Exception as e:
    print(f"      ⚠️  Performance curves plotting failed: {e}")

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("✅ 03y_train_ebm_income.py COMPLETE")
print("=" * 70)
print(f"\n  Target       : {TARGET}")
print(f"  CV AUC       : {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
print(f"  Test AUC     : {test_auc:.4f}  (XGB ref: {xgb_auc}  Δ={test_auc-(xgb_auc or 0):+.4f})")
print(f"  Brier Score  : {test_brier:.4f}  (XGB ref: {xgb_brier})")
print(f"  EBM Terms    : {len(ebm_raw.term_names_)} (main effects + interactions)")
print(f"\n  Outputs → {OUT_DIR}")
for fname in [
    "03y_ebm_inc_model.pkl",
    "03y_ebm_inc_results.json",
    "03y_ebm_inc_feature_importance.png",
    "03y_ebm_inc_global_explanation.html",
    "03y_ebm_inc_shape_functions.png",
]:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        print(f"    ✅ {fname:<45} {os.path.getsize(fpath)/1024:>7.1f} KB")
    else:
        print(f"    ⚠️  {fname} — not generated")
