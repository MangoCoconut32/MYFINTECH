"""
=============================================================================
02x_xgboost_calibrated.py — GIGA-BASELINE FOR PIPELINE X
=============================================================================
PURPOSE:
    Trains two Optuna-tuned, isotonically-calibrated XGBoost models
    (one per target) on the Master Dataset X (30 features).

    This is the calibrated benchmark that 03x (EBM) and 04x (TabNet)
    must beat to justify production routing to their respective targets.

ANTI-LEAKAGE:
    - All Optuna CV uses the pre-frozen fold column from Dataset_Needs_SOTA.csv.
    - CalibratedClassifierCV re-uses those same fold indices (cv=5).
    - X_test is never seen until the final evaluation block.

OUTPUTS (Output/Pipeline_X/):
    02x_xgb_{target}_calibrated.pkl    — CalibratedClassifierCV wrapper
    02x_performance_baseline.json      — AUC, Brier, P, R, F1 per target
    02x_calibration_curves.png         — Reliability diagrams (both targets)
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
from utilsx import (
    get_train_fold, get_full_train_val, get_test_set, get_cv_splitter,
    FEATURE_COLS, TARGET_COLS, RANDOM_STATE, PipelineXTransformer
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

PIPELINE_X_DIR = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_X")
# Parameters
N_TRIALS = 1
OUT_DIR  = PIPELINE_X_DIR
BEST_PARAMS_PATH = os.path.join(OUT_DIR, "02x_xgb_best_params.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Master Dataset X via Data Contract (utilsx)
# ---------------------------------------------------------------------------
print("=" * 70)
print("02x_xgboost_calibrated.py — Giga-Baseline XGBoost")
print("=" * 70)

print("\n[1/6] Loading RAW data via Pipeline X contract (stage='base')...")
X_tv_df, y_tv_df = get_full_train_val(stage="base")
X_test_df, y_test_df = get_test_set(stage="base")

# FEATURE_COLS will be defined by the first transform, 
# for now we use the ones from the Master stage as names reference.
# XGBoost will use all 15.
print(f"✅ Data Contract Verified: Using {len(FEATURE_COLS)} features")
print(f"      Train : {X_tv_df.shape[0]} samples")
print(f"      Test  : {X_test_df.shape[0]} samples")

# ---------------------------------------------------------------------------
# 2. Optuna — Two separate studies (one per target)
# ---------------------------------------------------------------------------
print(f"\n[2/6] Running Optuna ({N_TRIALS} trials × 2 targets) ...")

def make_objective(target_name):
    """Returns a closure that Optuna can optimize using frozen folds from utilsx."""
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
            # 1. Load specific fold through the contract (RAW STAGE)
            X_tr_fold, y_tr_fold, X_va_fold, y_va_fold = get_train_fold(i, stage="base")
            
            # 2. ANTI-LEAKAGE: Fit transformer ONLY on training folds
            transformer = PipelineXTransformer()
            transformer.fit(X_tr_fold)
            
            X_tr_eng = transformer.transform(X_tr_fold)
            X_va_eng = transformer.transform(X_va_fold)
            
            # Filter and strip ID
            X_tr = X_tr_eng[FEATURE_COLS].values
            y_tr = y_tr_fold[target_name].values
            X_va = X_va_eng[FEATURE_COLS].values
            y_va = y_va_fold[target_name].values
            
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            fold_aucs.append(roc_auc_score(y_va, probs))
            
        return float(np.mean(fold_aucs))
    return objective

best_params_per_target = {}

# Load existing best params for warm start if they exist
old_best_params = {}
if os.path.exists(BEST_PARAMS_PATH):
    try:
        with open(BEST_PARAMS_PATH, "r") as f:
            old_best_params = json.load(f)
        print(f"ℹ️  Warm Start: Loaded prior best parameters from {os.path.basename(BEST_PARAMS_PATH)}")
    except Exception as e:
        print(f"⚠️  Warm Start: Could not load prior params: {e}")

for target in TARGET_COLS:
    print(f"\n      -> Optimizing {target}...")
    study = optuna.create_study(direction="maximize",
                                study_name=f"XGB_Opt_{target}")
    
    # Enqueue previous best trial if available
    if target in old_best_params:
        study.enqueue_trial(old_best_params[target])
        print(f"      -> Enqueued prior best trial for {target}")

    study.optimize(make_objective(target), n_trials=N_TRIALS, show_progress_bar=False)

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

    # ANTI-LEAKAGE: Refit transformer on the ENTIRE 4000-row block for production
    transformer_final = PipelineXTransformer()
    transformer_final.fit(X_tv_df)
    
    X_train_clean = transformer_final.transform(X_tv_df)[FEATURE_COLS].values
    y_train_clean = y_tv_df[target].values
    
    # CalibratedClassifierCV with the same frozen folds via CV splitter
    # Note: CalibratedClassifierCV handles internal folds, but to be 100% pure 
    # we'd need to use a Pipe. For now, this final refit gives the baseline.
    cv_splits = get_cv_splitter()
    calibrated = CalibratedClassifierCV(
        base_xgb,
        method   = "isotonic",
        cv       = cv_splits,
        ensemble = False
    )
    calibrated.fit(X_train_clean, y_train_clean)
    calibrated_models[target] = (calibrated, transformer_final)
    print(f"      {target}: calibrated ✅")

# ---------------------------------------------------------------------------
# 4. Final Evaluation on Blind Test Set
# ---------------------------------------------------------------------------
print("\n[4/6] Evaluating on blind Test Set...")

performance = {}
probs_for_plot = {}

for target in TARGET_COLS:
    y_test  = y_test_df[target].values
    model, trans = calibrated_models[target]
    
    X_test_clean = trans.transform(X_test_df)[FEATURE_COLS].values
    probs   = model.predict_proba(X_test_clean)[:, 1]
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
        "Recall":    round(rec,  4),
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
    pkl_path  = os.path.join(OUT_DIR, f"02x_xgb_{safe_name}_calibrated.pkl")
    tr_path   = os.path.join(OUT_DIR, f"02x_xgb_{safe_name}_transformer.pkl")
    
    model, trans = calibrated_models[target]
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    with open(tr_path, "wb") as f:
        pickle.dump(trans, f)
    print(f"      Saved model & transformer: {safe_name}")

# Save final performance metrics
json_path = os.path.join(OUT_DIR, "02x_performance_baseline.json")
with open(json_path, "w") as f:
    json.dump(performance, f, indent=2)

# Save best parameters for future warm starts
with open(BEST_PARAMS_PATH, "w") as f:
    json.dump(best_params_per_target, f, indent=2)

print(f"\n✅ Pipeline X Baseline results saved to: {OUT_DIR}")
print("=" * 70)
print(f"      Saved metrics: {os.path.basename(json_path)}")

# ---------------------------------------------------------------------------
# 6. Presentation Visuals — Plot 1 (PR Curve) & Plot 2 (Reliability)
# ---------------------------------------------------------------------------
print("\n[6/6] Generating presentation visuals...")

# --- 6a. Reliability Diagram (Plot 2) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Pipeline X — Calibration Accuracy (Reliability Diagram)\nXGBoost + Isotonic Calibration",
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
plot_path_rel = os.path.join(OUT_DIR, "02x_calibration_curves.png")
fig.savefig(plot_path_rel, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"      Saved: 02x_calibration_curves.png")

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
ax.set_title("Pipeline X — Precision-Recall Tradeoff\nHigh-Precision Targeting for Managed Wealth",
             fontsize=16, fontweight="bold", loc="left", pad=15)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", fontsize=10, frameon=False)
ax.grid(True, linestyle="--", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

plot_path_pr = os.path.join(OUT_DIR, "02x_precision_recall.png")
fig.savefig(plot_path_pr, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"      Saved: 02x_precision_recall.png")

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("✅ 02x_xgboost_calibrated.py COMPLETE")
print("=" * 70)

summary_rows = []
for target, metrics in performance.items():
    summary_rows.append({"Target": target, **metrics})
summary_df = pd.DataFrame(summary_rows)
print("\n" + summary_df.to_string(index=False))

print(f"\nOutputs written to: {OUT_DIR}")
print("\n  📌 Benchmark locked. 03x (EBM) and 04x (TabNet) must beat these scores.")
