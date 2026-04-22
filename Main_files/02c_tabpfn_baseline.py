"""TabPFN zero-shot baseline.

The transformer was pre-trained on synthetic tabular tasks; at inference
you just hand it (X_train, y_train, X_test) and it returns probabilities
in one forward pass. No HP tuning, no scaling, raw features only -
otherwise it's not really zero-shot. Dataset is ~4000 rows so we're well
under the 10k cap.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_and_prepare_data

from tabpfn import TabPFNClassifier

_script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_script_dir, "..", "Dataset2_Needs.xls"))
OUT_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "02_baselines"))
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
for target in ["AccumulationInvestment", "IncomeInvestment"]:
    print(f"\n--- TARGET: {target} ---")

    X_train, X_test, y_train, y_test = load_and_prepare_data(
        FILE_PATH, target, use_engineered_features=False
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    clf = TabPFNClassifier()
    print("  Fitting TabPFN (single forward pass)...")
    clf.fit(X_train.values, y_train.values)

    print("  Predicting...")
    y_prob = clf.predict_proba(X_test.values)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    p = precision_score(y_test, y_pred)
    r = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    print(f"  AUC={auc:.4f}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")
    rows.append({
        "Target": target, "Model": "TabPFN (zero-shot)",
        "Test_ROC_AUC": round(auc, 4),
        "Test_Precision": round(p, 4),
        "Test_Recall": round(r, 4),
        "Test_F1": round(f, 4),
    })

df = pd.DataFrame(rows)
out = os.path.join(OUT_DIR, "02c_tabpfn_results.csv")
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(df.to_string(index=False))
