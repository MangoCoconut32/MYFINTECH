"""
=============================================================================
02y_xgboost_calibrated.py — GIGA-BASELINE FOR PIPELINE X
=============================================================================
PURPOSE:
    Trains two Optuna-tuned, isotonically-calibrated XGBoost models
    (one per target) on the Master Dataset Y (30 features).

    This is the calibrated benchmark that 03y (EBM) and 04y (TabNet)
    must beat to justify production routing to their respective targets.

ANTI-LEAKAGE:
    - All Optuna CV uses the pre-frozen fold column from Dataset_Needs_SOTA.csv.
    - CalibratedClassifierCV re-uses those same fold indices (cv=5).
    - X_test is never seen until the final evaluation block.

OUTPUTS (Output/Pipeline_Y/):
    02y_xgb_{target}_calibrated.pkl    — CalibratedClassifierCV wrapper
    02y_performance_baseline.json      — AUC, Brier, P, R, F1 per target
    02y_calibration_curves.png         — Reliability diagrams (both targets)
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

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
from utilsy import (
    get_train_fold, get_full_train_val, get_test_set, get_cv_splitter,
    FEATURE_COLS, TARGET_COLS, RANDOM_STATE
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

PIPELINE_X_DIR = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_Y")
# Parameters
N_TRIALS = 5
OUT_DIR  = PIPELINE_X_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Master Dataset Y via Data Contract (utilsy)
# ---------------------------------------------------------------------------
print("=" * 70)
print("02y_xgboost_calibrated.py — Giga-Baseline XGBoost")
print("=" * 70)

print("\n[1/6] Loading data via Pipeline Y contract (utilsy)...")
X_tv_df, y_tv_df = get_full_train_val()
X_test_df, y_test_df = get_test_set()

# We work with numpy arrays for XGBoost performance
X_train = X_tv_df[FEATURE_COLS].values
X_test  = X_test_df[FEATURE_COLS].values

print(f"✅ Data Contract Verified: Using {len(FEATURE_COLS)} features")
print(f"      Train : {X_train.shape}")
print(f"      Test  : {X_test.shape}")
print(f"      Features: {FEATURE_COLS[:5]} ...")

# ---------------------------------------------------------------------------
# 2. Optuna — Two separate studies (one per target)
# ---------------------------------------------------------------------------
print(f"\n[2/6] Running Optuna ({N_TRIALS} trials × 2 targets) ...")

def make_objective(target_name):
    """Returns a closure that Optuna can optimize using frozen folds from utilsy."""
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 800, step=100),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth         = trial.suggest_int("max_depth", 3, 8),
            min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
            subsample         = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma             = trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            eval_metric       = "logloss",
            tree_method       = "hist",
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            verbosity         = 0,
        )
        model = XGBClassifier(**params)
        fold_aucs = []
        
        for i in range(5):
            # Load specific fold through the contract
            X_tr_df, y_tr_df, X_va_df, y_va_df = get_train_fold(i)
            
            X_tr = X_tr_df[FEATURE_COLS].values
            y_tr = y_tr_df[target_name].values
            X_va = X_va_df[FEATURE_COLS].values
            y_va = y_va_df[target_name].values
            
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            fold_aucs.append(roc_auc_score(y_va, probs))
            
        return float(np.mean(fold_aucs))
    return objective

best_params_per_target = {}

for target in TARGET_COLS:
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(make_objective(target),
                   n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    best_auc = study.best_value
    print(f"        Best 5-Fold AUC : {best_auc:.4f}")
    print(f"        Best params     : {best}")
    best_params_per_target[target] = best

# ---------------------------------------------------------------------------
# 3. Calibration — CalibratedClassifierCV (isotonic, cv=5)
# ---------------------------------------------------------------------------
print("\n[3/6] Fitting CalibratedClassifierCV (isotonic)...")

calibrated_models = {}

for target in TARGET_COLS:
    bp = best_params_per_target[target]
    print(f"\n      Target: {target}")

    base_xgb = XGBClassifier(
        **bp,
        eval_metric       = "logloss",
        tree_method       = "hist",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbosity         = 0,
    )

    # CalibratedClassifierCV with the same frozen folds via CV splitter
    cv_splits = get_cv_splitter()
    calibrated = CalibratedClassifierCV(
        base_xgb,
        method   = "isotonic",
        cv       = cv_splits,         # frozen folds = zero leakage
        ensemble = False             # fits one model per fold, more efficient than ensemble=True which requires bagging
    )
    # Fit using full train-val block from utilsy
    calibrated.fit(X_train, y_tv_df[target].values)
    calibrated_models[target] = calibrated
    print(f"      {target}: calibrated ✅")

# ---------------------------------------------------------------------------
# 4. Final Evaluation on Blind Test Set
# ---------------------------------------------------------------------------
print("\n[4/6] Evaluating on blind Test Set...")

performance = {}
probs_for_plot = {}

for target in TARGET_COLS:
    y_test  = y_test_df[target].values
    model   = calibrated_models[target]
    probs   = model.predict_proba(X_test)[:, 1]
    preds   = (probs >= 0.5).astype(int)

    auc     = roc_auc_score(y_test, probs)
    brier   = brier_score_loss(y_test, probs)
    prec    = precision_score(y_test, preds, zero_division=0)
    rec     = recall_score(y_test, preds, zero_division=0)
    f1      = f1_score(y_test, preds, zero_division=0)

    performance[target] = {
        "AUC":       round(auc,   4),
        "Brier":     round(brier, 4),
        "Precision": round(prec,  4),
        "Recall":    round(rec,   4),
        "F1":        round(f1,    4),
    }
    probs_for_plot[target] = (y_test, probs)

    print(f"\n      {target}")
    print(f"        AUC           : {auc:.4f}")
    print(f"        Brier Score   : {brier:.4f}  (lower = more calibrated)")
    print(f"        Precision     : {prec:.4f}")
    print(f"        Recall        : {rec:.4f}")
    print(f"        F1            : {f1:.4f}")

# ---------------------------------------------------------------------------
# 5. Save models & JSON
# ---------------------------------------------------------------------------
print("\n[5/6] Saving models and metrics...")

for target in TARGET_COLS:
    safe_name = target.replace("Investment", "").lower()[:3]   # acc / inc
    pkl_path  = os.path.join(OUT_DIR, f"02y_xgb_{safe_name}_calibrated.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(calibrated_models[target], f)
    print(f"      Saved model: {os.path.basename(pkl_path)}")

json_path = os.path.join(OUT_DIR, "02y_performance_baseline.json")
with open(json_path, "w") as f:
    json.dump(performance, f, indent=2)
print(f"      Saved metrics: {os.path.basename(json_path)}")

# ---------------------------------------------------------------------------
# 6. Presentation Visuals — Plot 1 (PR Curve) & Plot 2 (Reliability)
# ---------------------------------------------------------------------------
print("\n[6/6] Generating presentation visuals...")

# --- 6a. Reliability Diagram (Plot 2) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Pipeline Y — Calibration Accuracy (Reliability Diagram)\nXGBoost + Isotonic Calibration",
             fontweight="bold", fontsize=16, x=0.05, ha="left", y=1.02)

COLORS = {"AccumulationInvestment": "#1B3A6B", "IncomeInvestment": "#2E86C1"}

for ax, target in zip(axes, TARGET_COLS):
    y_true, y_prob = probs_for_plot[target]

    # Reliability diagram (10 bins)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, "s-", color=COLORS[target],
            label="Calibrated XGB", linewidth=2.5, markersize=8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Ideal Calibration")

    # Histogram (predicted probability distribution)
    ax2 = ax.twinx()
    ax2.hist(y_prob, bins=25, alpha=0.15, color=COLORS[target])
    ax2.set_ylabel("Density / Count", fontsize=9, color="#AEB6BF")
    ax2.tick_params(axis="y", labelsize=8)

    perf = performance[target]
    ax.set_title(f"{target}\nAUC={perf['AUC']} | Brier={perf['Brier']}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives", fontsize=10)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plot_path_rel = os.path.join(OUT_DIR, "02y_calibration_curves.png")
fig.savefig(plot_path_rel, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"      Saved: 02y_calibration_curves.png")

# --- 6b. Precision-Recall Curve (Plot 1) ---
fig, ax = plt.subplots(figsize=(10, 6))
for target in TARGET_COLS:
    y_true, y_prob = probs_for_plot[target]
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_p = average_precision_score(y_true, y_prob)
    
    ax.plot(recall, precision, color=COLORS[target], linewidth=3,
            label=f"{target} (AP={avg_p:.3f})")

# Target indicator: highlighting 0.90 Precision for Income
ax.axhline(0.90, color="#E74C3C", linestyle="--", alpha=0.6, label="Business Target: 90% Precision")
ax.fill_between([0, 1], 0.90, 1.0, color="#E74C3C", alpha=0.05)

ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
ax.set_ylabel("Precision (Exactness)", fontsize=11)
ax.set_title("Pipeline Y — Precision-Recall Tradeoff\nHigh-Precision Targeting for Managed Wealth",
             fontsize=16, fontweight="bold", loc="left", pad=15)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", fontsize=10, frameon=False)
ax.grid(True, linestyle="--", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

plot_path_pr = os.path.join(OUT_DIR, "02y_precision_recall.png")
fig.savefig(plot_path_pr, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"      Saved: 02y_precision_recall.png")

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("✅ 02y_xgboost_calibrated.py COMPLETE")
print("=" * 70)

summary_rows = []
for target, metrics in performance.items():
    summary_rows.append({"Target": target, **metrics})
summary_df = pd.DataFrame(summary_rows)
print("\n" + summary_df.to_string(index=False))

print(f"\nOutputs written to: {OUT_DIR}")
print("\n  📌 Benchmark locked. 03y (EBM) and 04y (TabNet) must beat these scores.")
