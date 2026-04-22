"""Significance tests for the headline AUC numbers.

Two things in here:
  - 95% bootstrap CI on test AUC per model (n=2000 resamples)
  - Paired test (bootstrap-based, DeLong-style) on Optuna XGB vs the
    calibrated variant - we want to know if calibration actually moved
    the AUC or not

Used to back up the claims in report.md.
"""
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_and_prepare_data

_script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_script_dir, "..", "Dataset2_Needs.xls"))
MODEL_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "04_optuna"))
OUT_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "09_statistical_significance"))
os.makedirs(OUT_DIR, exist_ok=True)


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_roc_variance(y_true, y_score):
    order = np.argsort(-y_score)
    label_1_count = int(y_true.sum())
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]
    pos = y_score_sorted[y_true_sorted == 1]
    neg = y_score_sorted[y_true_sorted == 0]
    return pos, neg


def delong_test(y_true, score_a, score_b):
    """Two-sided DeLong test for AUC(A) == AUC(B) on paired samples."""
    y_true = np.asarray(y_true).astype(int)
    auc_a = roc_auc_score(y_true, score_a)
    auc_b = roc_auc_score(y_true, score_b)


    rng = np.random.default_rng(42)
    n = len(y_true)
    diffs = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        diffs.append(roc_auc_score(yt, score_a[idx]) - roc_auc_score(yt, score_b[idx]))
    diffs = np.array(diffs)
    se = diffs.std(ddof=1)
    z = (auc_a - auc_b) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, auc_a - auc_b, se, z, p


def bootstrap_auc_ci(y_true, y_score, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        aucs.append(roc_auc_score(yt, y_score[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return np.mean(aucs), lo, hi


rows = []
for target in ["AccumulationInvestment", "IncomeInvestment"]:
    print(f"\n--- {target} ---")
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        FILE_PATH, target, use_engineered_features=True
    )

    with open(f"{MODEL_DIR}/04_optuna_{target}_xgb.pkl", "rb") as f:
        xgb_raw = pickle.load(f)
    xgb_cal = joblib.load(f"{MODEL_DIR}/04b_{target}_xgb_calibrated.pkl")

    p_raw = xgb_raw.predict_proba(X_test)[:, 1]
    p_cal = xgb_cal.predict_proba(X_test)[:, 1]


    m_raw, lo_raw, hi_raw = bootstrap_auc_ci(y_test.values, p_raw)
    m_cal, lo_cal, hi_cal = bootstrap_auc_ci(y_test.values, p_cal)
    print(f"  XGB Optuna        AUC = {m_raw:.4f}  95% CI [{lo_raw:.4f}, {hi_raw:.4f}]")
    print(f"  XGB Calibrated    AUC = {m_cal:.4f}  95% CI [{lo_cal:.4f}, {hi_cal:.4f}]")


    a, b, diff, se, z, p = delong_test(y_test.values, p_raw, p_cal)
    print(f"  Paired test: ΔAUC = {diff:+.4f}  SE = {se:.4f}  z = {z:+.2f}  p = {p:.3f}")

    rows.append({
        "Target": target,
        "XGB_AUC": round(m_raw, 4), "XGB_CI_lo": round(lo_raw, 4), "XGB_CI_hi": round(hi_raw, 4),
        "Calib_AUC": round(m_cal, 4), "Calib_CI_lo": round(lo_cal, 4), "Calib_CI_hi": round(hi_cal, 4),
        "Delta_AUC": round(diff, 4), "SE": round(se, 4), "z": round(z, 3), "p_value": round(p, 4),
        "significant_at_0.05": p < 0.05,
    })

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/09_significance_results.csv", index=False)
print(f"\nSaved: {OUT_DIR}/09_significance_results.csv")
print(df.to_string(index=False))
