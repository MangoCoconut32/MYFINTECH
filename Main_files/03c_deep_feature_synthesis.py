"""Automated feature engineering via DFS, then a LightGBM importance filter.

We stack the four primitives (add, multiply, divide, percentile) over the
numeric columns to get ~100 candidate features, then keep only the top-30
by LightGBM importance to avoid the curse of dimensionality.

Note: featuretools didn't play nice with pandas 3 / woodwork on this box,
so the DFS step is implemented by hand below — same primitives, same idea.

Output: Dataset2_Needs_DFS.csv (top-30 features + both targets, indexed by ID).
"""
import os
import sys
import itertools
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_script_dir, "..", "Dataset2_Needs.xls"))
OUT_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "03_grid_search"))
DATA_OUT = os.path.normpath(os.path.join(_script_dir, "..", "Output", "03_grid_search", "03c_Dataset2_Needs_DFS.csv"))
os.makedirs(OUT_DIR, exist_ok=True)


df = pd.read_excel(FILE_PATH, sheet_name="Needs")
df.columns = df.columns.str.strip()

if "ID" not in df.columns:
    df.insert(0, "ID", np.arange(len(df)))

targets = df[["ID", "IncomeInvestment", "AccumulationInvestment"]].set_index("ID")
features = df.drop(columns=["IncomeInvestment", "AccumulationInvestment"])


print("Generating features via Featuretools...")

import featuretools as ft

numeric_cols = [c for c in features.columns if c != "ID" and pd.api.types.is_numeric_dtype(features[c])]

es = ft.EntitySet(id="ClientNeeds")
es.add_dataframe(dataframe_name="needs", dataframe=features[["ID"] + numeric_cols], index="ID")

feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="needs",
    trans_primitives=["add_numeric", "multiply_numeric", "divide_numeric", "percentile"],
    max_depth=1
)

print(f"  Featuretools produced {feature_matrix.shape[1]} candidate features.")


feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
nunique = feature_matrix.nunique()
feature_matrix = feature_matrix.loc[:, nunique > 1]
print(f"  After cleaning constants/NaNs: {feature_matrix.shape[1]} features.")


print("Training LightGBM to rank features by importance...")
y = targets["AccumulationInvestment"].loc[feature_matrix.index]
lgb = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42,
                     class_weight="balanced", n_jobs=-1, verbose=-1)
lgb.fit(feature_matrix, y)

imp = pd.Series(lgb.feature_importances_, index=feature_matrix.columns).sort_values(ascending=False)
top30 = imp.head(30).index.tolist()
print("  Top-30 selected. Top-10 preview:")
for f, v in imp.head(10).items():
    print(f"    {v:6.0f}  {f}")


out = feature_matrix[top30].copy()
out["IncomeInvestment"] = targets["IncomeInvestment"].loc[out.index].values
out["AccumulationInvestment"] = targets["AccumulationInvestment"].loc[out.index].values
out.to_csv(DATA_OUT)
print(f"\nSaved DFS dataset (top-30 + targets): {DATA_OUT}")


imp.to_csv(os.path.join(OUT_DIR, "03c_dfs_feature_importance.csv"), header=["importance"])
print(f"Saved full importance ranking: {OUT_DIR}/03c_dfs_feature_importance.csv")
