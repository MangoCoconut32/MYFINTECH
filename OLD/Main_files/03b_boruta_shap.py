"""Feature selection using Boruta-SHAP.

Boruta wraps a tree model and adds shuffled "shadow" copies of every
column; a real feature is only accepted if its importance beats all of
the shadows. Using SHAP for importance instead of the default Gini -
less biased toward high-cardinality columns.

Output: JSON of accepted / rejected / tentative features per target.
Other scripts can pick this up to drop noise columns before training.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_and_prepare_data

import scipy.stats as _stats
if not hasattr(_stats, "binom_test"):
    def binom_test(x, n=None, p=0.5, alternative="two-sided"):
        return _stats.binomtest(int(x), n=int(n), p=p, alternative=alternative).pvalue
    _stats.binom_test = binom_test
if not hasattr(np, "NaN"):
    np.NaN = np.nan

from BorutaShap import BorutaShap

_script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_script_dir, "..", "Dataset2_Needs.xls"))
OUT_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "03_grid_search"))
os.makedirs(OUT_DIR, exist_ok=True)

results = {}
for target in ["AccumulationInvestment", "IncomeInvestment"]:
    print(f"\n--- TARGET: {target} ---")
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        FILE_PATH, target, use_engineered_features=True
    )

    base = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )

    selector = BorutaShap(
        model=base,
        importance_measure="shap",
        classification=True,
    )
    print("  Running Boruta-SHAP (50 trials)...")
    selector.fit(X=X_train, y=y_train, n_trials=50, random_state=42, verbose=False)

    accepted = sorted(list(selector.accepted))
    rejected = sorted(list(selector.rejected))
    tentative = sorted(list(selector.tentative))

    print(f"  Accepted ({len(accepted)}):  {accepted}")
    print(f"  Tentative ({len(tentative)}): {tentative}")
    print(f"  Rejected ({len(rejected)}):  {rejected}")

    results[target] = {
        "accepted": accepted,
        "tentative": tentative,
        "rejected": rejected,
        "n_features_total": int(X_train.shape[1]),
    }

out = os.path.join(OUT_DIR, "03b_boruta_selected_features.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
