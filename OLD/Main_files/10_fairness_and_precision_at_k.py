"""Fairness audit + Precision@K for the recommender.

Two evaluations:

  1. AUC per slice (Age bracket, Gender, FamilyMembers cohort) for the
     Optuna XGB. Useful for the MIFID "target-market suitability" angle -
     we want to flag any subgroup the model under-serves.
  2. Precision@K / Recall@K on the per-client ranking of the two products,
     using the Needs flags as the relevance ground truth.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_and_prepare_data

_script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_script_dir, "..", "Dataset2_Needs.xls"))
MODEL_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "04_optuna"))
OUT_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "10_fairness_precision_at_k"))
os.makedirs(OUT_DIR, exist_ok=True)


def slice_auc(X_test, y_test, y_prob, slice_col, slice_fn):
    """Compute AUC for each slice produced by slice_fn applied to X_test."""
    out = []
    groups = slice_fn(X_test)
    for g, mask in groups.items():
        if mask.sum() < 20:
            continue
        if y_test[mask].nunique() < 2:
            continue
        out.append({
            "slice_variable": slice_col,
            "slice": g,
            "n": int(mask.sum()),
            "positive_rate": round(float(y_test[mask].mean()), 3),
            "AUC": round(roc_auc_score(y_test[mask], y_prob[mask]), 4),
        })
    return out


def age_slices(X):
    return {
        "Young (18-35)": X.get("Age_bracket_Young", 0).astype(bool)
                         if "Age_bracket_Young" in X.columns else (X["Age"] <= 35),
        "Mid (36-55)":   X.get("Age_bracket_Mid", 0).astype(bool)
                         if "Age_bracket_Mid" in X.columns else ((X["Age"] > 35) & (X["Age"] <= 55)),
        "Senior (55+)":  X.get("Age_bracket_Senior", 0).astype(bool)
                         if "Age_bracket_Senior" in X.columns else (X["Age"] > 55),
    }


def gender_slices(X):
    return {"Gender=0": X["Gender"] == 0, "Gender=1": X["Gender"] == 1}


def family_slices(X):
    return {
        "Singleton (FM=1)":    X["FamilyMembers"] == 1,
        "Small (FM=2-3)":      X["FamilyMembers"].between(2, 3),
        "Large (FM>=4)":       X["FamilyMembers"] >= 4,
    }


rows = []
for target in ["AccumulationInvestment", "IncomeInvestment"]:
    print(f"\n--- Fairness slices: {target} ---")
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        FILE_PATH, target, use_engineered_features=True
    )
    with open(f"{MODEL_DIR}/04_optuna_{target}_xgb.pkl", "rb") as f:
        xgb = pickle.load(f)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    base_auc = roc_auc_score(y_test, y_prob)
    print(f"  Overall AUC: {base_auc:.4f}")

    all_slices = []
    all_slices += slice_auc(X_test, y_test, y_prob, "Age", age_slices)
    all_slices += slice_auc(X_test, y_test, y_prob, "Gender", gender_slices)
    all_slices += slice_auc(X_test, y_test, y_prob, "FamilyMembers", family_slices)
    for r in all_slices:
        r["Target"] = target
        r["Overall_AUC"] = round(base_auc, 4)
        r["Delta_vs_overall"] = round(r["AUC"] - base_auc, 4)
        rows.append(r)
        print(f"    {r['slice_variable']:14s} | {r['slice']:20s} | n={r['n']:4d} | AUC={r['AUC']:.4f} "
              f"| Δ={r['Delta_vs_overall']:+.4f}")

fair_df = pd.DataFrame(rows)
fair_df.to_csv(f"{OUT_DIR}/10_fairness_slices.csv", index=False)
print(f"\nSaved: {OUT_DIR}/10_fairness_slices.csv")


print("\n--- Precision@K for two-product recommender ---")

X_train_a, X_test_a, y_train_a, y_test_a = load_and_prepare_data(
    FILE_PATH, "AccumulationInvestment", use_engineered_features=True
)
full = pd.read_excel(FILE_PATH, sheet_name="Needs")
full.columns = full.columns.str.strip()
y_test_i = full.loc[X_test_a.index, "IncomeInvestment"]

with open(f"{MODEL_DIR}/04_optuna_AccumulationInvestment_xgb.pkl", "rb") as f:
    xgb_a = pickle.load(f)
with open(f"{MODEL_DIR}/04_optuna_IncomeInvestment_xgb.pkl", "rb") as f:
    xgb_i = pickle.load(f)


_, X_test_i, _, _ = load_and_prepare_data(FILE_PATH, "IncomeInvestment", use_engineered_features=True)
p_a = xgb_a.predict_proba(X_test_a)[:, 1]
p_i = xgb_i.predict_proba(X_test_i)[:, 1]


scores = pd.DataFrame({"Accumulation": p_a, "Income": p_i}, index=X_test_a.index)
relevance = pd.DataFrame(
    {"Accumulation": y_test_a.values, "Income": y_test_i.values}, index=X_test_a.index
)

def precision_at_k(scores, relevance, k):
    ranked = scores.rank(axis=1, ascending=False, method="first")
    top_k_mask = ranked <= k
    hits = (top_k_mask & (relevance == 1)).sum(axis=1)
    rel_counts = relevance.sum(axis=1).clip(lower=1)

    precision = (hits / k).mean()
    recall = (hits / rel_counts).mean()
    return precision, recall

p_rows = []
for k in (1, 2):
    p, r = precision_at_k(scores, relevance, k)
    p_rows.append({"K": k, "Precision@K": round(p, 4), "Recall@K": round(r, 4)})
    print(f"  K={k}  Precision@K = {p:.4f}  Recall@K = {r:.4f}")

pd.DataFrame(p_rows).to_csv(f"{OUT_DIR}/10_precision_at_k.csv", index=False)
print(f"\nSaved: {OUT_DIR}/10_precision_at_k.csv")
