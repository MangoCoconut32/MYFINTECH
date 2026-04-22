"""CatBoost + LightGBM baselines next to the canonical XGBoost.

Trying to see if either of these beats the XGB ceiling on the tabular data.
- CatBoost: skip OHE, pass Gender / Age_bracket via cat_features.
- LightGBM: same engineered features but with class_weight='balanced',
  since IncomeInvestment is the imbalanced one (XGB tops out around 0.76).
- XGB is re-evaluated as the reference.

Output: Output/02_baselines/02b_catboost_lgbm_results.csv
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabulate import tabulate

script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH  = os.path.normpath(os.path.join(script_dir, "..", "Dataset2_Needs.xls"))
OUT_DIR    = os.path.normpath(os.path.join(script_dir, "..", "Output", "02_baselines"))
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(FILE_PATH):
    print("Error: Could not find Dataset2_Needs.xls.")
    sys.exit(1)

TARGETS = ["AccumulationInvestment", "IncomeInvestment"]


def load_with_categoricals(target_col):
    """
    Return (X_train, X_test, y_train, y_test, cat_feature_names).
    Mirrors utils.load_and_prepare_data feature engineering, but keeps
    Gender and Age_bracket as explicit categorical columns rather than
    one-hot expanding them. Numerical scaling is skipped - GBDTs are
    scale-invariant.
    """
    df = pd.read_excel(FILE_PATH, sheet_name="Needs")
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["ID"])

    y = df[target_col]
    X = df.drop(columns=["IncomeInvestment", "AccumulationInvestment"])


    X["Wealth_log"]          = np.log1p(X["Wealth"])
    X["Income_log"]          = np.log1p(X["Income"])
    X["Wealth_per_person"]   = X["Wealth"] / X["FamilyMembers"]
    X["Income_per_person"]   = X["Income"] / X["FamilyMembers"]
    X["Inc_to_Wealth_ratio"] = X["Income"].div(X["Wealth"].replace(0, np.nan))
    X["Inc_to_Wealth_ratio"] = X["Inc_to_Wealth_ratio"].fillna(X["Income"].max())


    X["Age_bracket"] = pd.cut(
        X["Age"], bins=[17, 35, 55, 100], labels=["Young", "Mid", "Senior"]
    ).astype(str)
    X["Gender"] = X["Gender"].astype(str)

    cat_features = ["Gender", "Age_bracket"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, cat_features


def evaluate(model, X_train, X_test, y_train, y_test, fit_kwargs=None, manual_cv=False):
    """5-fold CV on train + final metrics on test.

    CatBoost's `cat_features` breaks sklearn.clone, so we use a manual CV
    loop that constructs a fresh estimator per fold via reflection.
    """
    fit_kwargs = fit_kwargs or {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if manual_cv:
        aucs = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            m = type(model)(**model.get_params())
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], **fit_kwargs)
            p = m.predict_proba(X_train.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[val_idx], p))
        cv_auc = float(np.mean(aucs))
    else:
        cv_auc = cross_val_score(model, X_train, y_train, cv=skf,
                                 scoring="roc_auc", n_jobs=-1).mean()

    model.fit(X_train, y_train, **fit_kwargs)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "cv_auc":   cv_auc,
        "test_auc": roc_auc_score(y_test, y_prob),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall":    recall_score(y_test, y_pred, zero_division=0),
        "test_f1":        f1_score(y_test, y_pred, zero_division=0),
    }


print("=" * 100)
print("STEP 02b: CATBOOST + LIGHTGBM vs XGBOOST BASELINE")
print("=" * 100)

rows = []

for target in TARGETS:
    print(f"\n--- TARGET: {target} ---")
    X_tr, X_te, y_tr, y_te, cat_features = load_with_categoricals(target)

    print("[1] CatBoost (native cat_features, no OHE)...")
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
    )
    r = evaluate(cat_model, X_tr, X_te, y_tr, y_te, manual_cv=True)
    rows.append({"Target": target, "Model": "CatBoost (native cat)", **r})

    print("[2] LightGBM (class_weight='balanced')...")
    X_tr_lgb = X_tr.copy(); X_te_lgb = X_te.copy()
    for c in cat_features:
        X_tr_lgb[c] = X_tr_lgb[c].astype("category")
        X_te_lgb[c] = X_te_lgb[c].astype("category")
    lgb_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    r = evaluate(lgb_model, X_tr_lgb, X_te_lgb, y_tr, y_te)
    rows.append({"Target": target, "Model": "LightGBM (balanced)", **r})


    print("[3] XGBoost reference (OHE of cat features for fair compare)...")
    X_tr_xgb = pd.get_dummies(X_tr, columns=cat_features, drop_first=False, dtype=int)
    X_te_xgb = pd.get_dummies(X_te, columns=cat_features, drop_first=False, dtype=int)

    X_te_xgb = X_te_xgb.reindex(columns=X_tr_xgb.columns, fill_value=0)
    xgb_model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        eval_metric="logloss", random_state=42,
    )
    r = evaluate(xgb_model, X_tr_xgb, X_te_xgb, y_tr, y_te)
    rows.append({"Target": target, "Model": "XGBoost (reference)", **r})


df = pd.DataFrame(rows)
df = df.round(3)
csv_path = os.path.join(OUT_DIR, "02b_catboost_lgbm_results.csv")
df.to_csv(csv_path, index=False)

print("\n" + "=" * 100)
print("STEP 02b: RESULTS")
print("=" * 100)
print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
print(f"\nSaved: {csv_path}")
