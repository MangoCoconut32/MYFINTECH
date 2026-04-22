"""DiCE counterfactual explanations on top of the XGB model.

SHAP says *why* a client got a given prediction; DiCE says *what would
have to change* for the prediction to flip. We only let the actionable
columns vary (wealth, income, financial education / risk scores) - Age,
Gender and FamilyMembers are locked.
"""
import os
import pickle
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_and_prepare_data

import dice_ml
from dice_ml import Dice

_script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_script_dir, "..", "Dataset2_Needs.xls"))
MODEL_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "04_optuna"))
OUT_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "07_XAI_Report"))
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "AccumulationInvestment"


X_train, X_test, y_train, y_test = load_and_prepare_data(
    FILE_PATH, TARGET, use_engineered_features=True
)
with open(f"{MODEL_DIR}/04_optuna_{TARGET}_xgb.pkl", "rb") as f:
    xgb = pickle.load(f)


train_df = X_train.copy()
train_df[TARGET] = y_train.values


IMMUTABLE = ["Age", "Gender", "FamilyMembers"] + [c for c in X_train.columns if c.startswith("Age_bracket_")]
MUTABLE = [c for c in X_train.columns if c not in IMMUTABLE]
CONTINUOUS = [c for c in MUTABLE if X_train[c].nunique() > 10]

d = dice_ml.Data(
    dataframe=train_df,
    continuous_features=CONTINUOUS,
    outcome_name=TARGET,
)
m = dice_ml.Model(model=xgb, backend="sklearn", model_type="classifier")
exp = Dice(d, m, method="random")


probs = xgb.predict_proba(X_test)[:, 1]
low_idx = np.argsort(probs)[:5]                                

print(f"Generating counterfactuals for 5 clients with lowest predicted prob...")
print(f"  Immutable features: {IMMUTABLE}")
print(f"  Mutable features:   {MUTABLE}")

query = X_test.iloc[low_idx].reset_index(drop=True)
cf = exp.generate_counterfactuals(
    query,
    total_CFs=3,
    desired_class=1,
    features_to_vary=MUTABLE,
)


all_results = []
for i, cf_ex in enumerate(cf.cf_examples_list):
    orig = query.iloc[[i]].copy()
    orig["_kind"] = "original"
    orig["_client_idx"] = int(low_idx[i])
    orig["_predicted_prob"] = probs[low_idx[i]]
    all_results.append(orig)
    if cf_ex.final_cfs_df is not None:
        cfs = cf_ex.final_cfs_df.copy()
        cfs["_kind"] = "counterfactual"
        cfs["_client_idx"] = int(low_idx[i])
        cfs["_predicted_prob"] = np.nan
        all_results.append(cfs)

results = pd.concat(all_results, ignore_index=True)
results.to_csv(f"{OUT_DIR}/07b_dice_counterfactuals.csv", index=False)

print(f"\nSaved: {OUT_DIR}/07b_dice_counterfactuals.csv")
print(f"Summary: {len(results[results['_kind']=='counterfactual'])} counterfactuals for "
      f"{results['_client_idx'].nunique()} clients.")
