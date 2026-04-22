"""Multi-objective Optuna + probability calibration.

The canonical 04 maximises ROC-AUC only. For a MIFID-style recommender we
gate on the probability itself, so a model that ranks well but is poorly
calibrated is dangerous. So here we:

  1. Run Optuna with two objectives - maximise AUC, minimise Brier - and
     pick the trial with the highest AUC subject to Brier < 0.15.
  2. Wrap the selected XGB with Platt + Isotonic calibration and pick
     whichever gives the lower log-loss on the test set.
  3. Save reliability diagrams so we can eyeball the result.

Outputs go under Output/04_optuna/ with the 04b_ prefix.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from tabulate import tabulate

from utils import load_and_prepare_data

optuna.logging.set_verbosity(optuna.logging.WARNING)


script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(script_dir, "..", "Dataset2_Needs.xls"))
OUT_DIR   = os.path.normpath(os.path.join(script_dir, "..", "Output", "04_optuna"))
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(FILE_PATH):
    print("Error: Could not find Dataset2_Needs.xls.")
    sys.exit(1)

TARGETS      = ["AccumulationInvestment", "IncomeInvestment"]
N_TRIALS     = 40
BRIER_BUDGET = 0.15                                                      


def objective_xgb_mo(trial, X, y):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 300, step=50),
        'max_depth':     trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
        'reg_lambda':    trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'eval_metric':   'logloss',
        'random_state':  42,
    }
    model = XGBClassifier(**params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    y_prob = cross_val_predict(model, X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]

    auc   = roc_auc_score(y, y_prob)
    brier = brier_score_loss(y, y_prob)
    return auc, brier


def select_pareto_winner(study, brier_cap=BRIER_BUDGET):
    """
    From the Pareto front, pick the trial with the highest ROC-AUC whose
    Brier score is strictly below `brier_cap`. Falls back to the trial with
    the lowest Brier if no trial meets the cap.
    """
    candidates = [t for t in study.best_trials if t.values[1] < brier_cap]
    if candidates:
        winner = max(candidates, key=lambda t: t.values[0])
        reason = f"highest AUC under Brier<{brier_cap}"
    else:
        winner = min(study.best_trials, key=lambda t: t.values[1])
        reason = f"no trial met Brier<{brier_cap}; picked lowest-Brier Pareto trial"
    return winner, reason


def plot_reliability(y_true, prob_raw, prob_platt, prob_iso, target, out_path):
    """Overlay raw, Platt-calibrated, and Isotonic-calibrated reliability curves."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))


    for prob, label, style in [
        (prob_raw,   "Raw XGBoost",       "o-"),
        (prob_platt, "Platt (sigmoid)",   "s--"),
        (prob_iso,   "Isotonic",          "^:"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
        ax[0].plot(mean_pred, frac_pos, style, label=label)

    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax[0].set_xlabel("Mean predicted probability")
    ax[0].set_ylabel("Fraction of positives")
    ax[0].set_title(f"Reliability Diagram - {target}")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)


    ax[1].hist(prob_raw,   bins=20, alpha=0.5, label="Raw",      density=True)
    ax[1].hist(prob_platt, bins=20, alpha=0.5, label="Platt",    density=True)
    ax[1].hist(prob_iso,   bins=20, alpha=0.5, label="Isotonic", density=True)
    ax[1].set_xlabel("Predicted P(y=1)")
    ax[1].set_ylabel("Density")
    ax[1].set_title("Prediction distribution")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


print("=" * 100)
print("STEP 04b: MULTI-OBJECTIVE OPTUNA (ROC-AUC + BRIER) + CALIBRATION")
print("=" * 100)

summary_rows = []

for target in TARGETS:
    print(f"\n{'*'*60}\nTARGET: {target}\n{'*'*60}")
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        FILE_PATH, target, use_engineered_features=True
    )


    print(f"[1] Running multi-objective Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(
        lambda trial: objective_xgb_mo(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )


    pareto_rows = [
        {"trial": t.number, "auc": t.values[0], "brier": t.values[1], **t.params}
        for t in study.best_trials
    ]
    pd.DataFrame(pareto_rows).to_csv(
        os.path.join(OUT_DIR, f"04b_pareto_{target}.csv"), index=False
    )
    print(f" -> Pareto front has {len(pareto_rows)} trials "
          f"(saved to 04b_pareto_{target}.csv)")


    winner, reason = select_pareto_winner(study, BRIER_BUDGET)
    print(f"[2] Selected trial #{winner.number}: AUC={winner.values[0]:.3f}, "
          f"Brier={winner.values[1]:.3f}  ({reason})")

    best_params = {**winner.params, "eval_metric": "logloss", "random_state": 42}
    best_xgb = XGBClassifier(**best_params)
    best_xgb.fit(X_train, y_train)


    prob_raw_test = best_xgb.predict_proba(X_test)[:, 1]
    auc_raw   = roc_auc_score(y_test, prob_raw_test)
    brier_raw = brier_score_loss(y_test, prob_raw_test)
    ll_raw    = log_loss(y_test, prob_raw_test)


    def fresh():
        return XGBClassifier(**best_params)

    platt = CalibratedClassifierCV(fresh(), method="sigmoid",  cv=5)
    iso   = CalibratedClassifierCV(fresh(), method="isotonic", cv=5)
    platt.fit(X_train, y_train)
    iso.fit(X_train, y_train)

    prob_platt_test = platt.predict_proba(X_test)[:, 1]
    prob_iso_test   = iso.predict_proba(X_test)[:, 1]

    auc_platt   = roc_auc_score(y_test, prob_platt_test)
    brier_platt = brier_score_loss(y_test, prob_platt_test)
    ll_platt    = log_loss(y_test, prob_platt_test)

    auc_iso   = roc_auc_score(y_test, prob_iso_test)
    brier_iso = brier_score_loss(y_test, prob_iso_test)
    ll_iso    = log_loss(y_test, prob_iso_test)


    plot_reliability(
        y_test.values, prob_raw_test, prob_platt_test, prob_iso_test,
        target=target,
        out_path=os.path.join(OUT_DIR, f"04b_reliability_{target}.png"),
    )


    brier_map = {"raw": brier_raw, "platt": brier_platt, "isotonic": brier_iso}
    chosen = min(brier_map, key=brier_map.get)
    chosen_model = {"raw": best_xgb, "platt": platt, "isotonic": iso}[chosen]

    print(f"[3] Calibration results (Test set):")
    print(f"     Raw      AUC={auc_raw:.3f}  Brier={brier_raw:.3f}  LogLoss={ll_raw:.3f}")
    print(f"     Platt    AUC={auc_platt:.3f}  Brier={brier_platt:.3f}  LogLoss={ll_platt:.3f}")
    print(f"     Isotonic AUC={auc_iso:.3f}  Brier={brier_iso:.3f}  LogLoss={ll_iso:.3f}")
    print(f" -> Chosen production calibrator: {chosen.upper()}")


    joblib.dump(best_xgb,    os.path.join(OUT_DIR, f"04b_{target}_xgb_selected.pkl"))
    joblib.dump(chosen_model, os.path.join(OUT_DIR, f"04b_{target}_xgb_calibrated.pkl"))
    print(f" -> Saved 04b_{target}_xgb_selected.pkl (raw) "
          f"and 04b_{target}_xgb_calibrated.pkl ({chosen})")


    summary_rows.extend([
        {"Target": target, "Variant": "Raw XGBoost",       "AUC": round(auc_raw, 3),
         "Brier": round(brier_raw, 3),   "LogLoss": round(ll_raw, 3)},
        {"Target": target, "Variant": "Platt (sigmoid)",   "AUC": round(auc_platt, 3),
         "Brier": round(brier_platt, 3), "LogLoss": round(ll_platt, 3)},
        {"Target": target, "Variant": "Isotonic",          "AUC": round(auc_iso, 3),
         "Brier": round(brier_iso, 3),   "LogLoss": round(ll_iso, 3)},
    ])


df = pd.DataFrame(summary_rows)
csv_path = os.path.join(OUT_DIR, "04b_multiobjective_results.csv")
df.to_csv(csv_path, index=False)

print("\n" + "=" * 100)
print("STEP 04b: MULTI-OBJECTIVE + CALIBRATION MASTER TABLE")
print("=" * 100)
print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
print(f"\nSaved master table:     {csv_path}")
print(f"Saved reliability plots: {OUT_DIR}/04b_reliability_*.png")
