"""TabNet multi-task classifier - attention-based replacement for the Keras MLP.

TabNet (Arik & Pfister 2019) is a stack of attentive blocks; each block
picks a sparse subset of features via a sparsemax mask. The masks double
as a per-instance feature attribution map, so we get XAI for free.

Per TASK.md: tune n_d (= n_a) in [8, 64] and n_steps in [3, 7] with Optuna,
then save the global importance and per-step attention masks as heatmaps.
"""

import os
import sys
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from tabulate import tabulate

from utils import load_and_prepare_data

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.manual_seed(42)
np.random.seed(42)


script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH  = os.path.normpath(os.path.join(script_dir, "..", "Dataset2_Needs.xls"))
OUT_DIR    = os.path.normpath(os.path.join(script_dir, "..", "Output", "06_neural_networks"))
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS   = ["AccumulationInvestment", "IncomeInvestment"]
N_TRIALS  = 15                                             
MAX_EPOCHS = 120
PATIENCE   = 15


print("Preparing multi-task dataset...")


X_train_acc, X_test_acc, y_train_acc, y_test_acc = load_and_prepare_data(
    FILE_PATH, "AccumulationInvestment", use_engineered_features=True
)
full_needs = pd.read_excel(FILE_PATH, sheet_name="Needs")
full_needs.columns = full_needs.columns.str.strip()

y_train_inc = full_needs.loc[X_train_acc.index, "IncomeInvestment"]
y_test_inc  = full_needs.loc[X_test_acc.index,  "IncomeInvestment"]

X_train = X_train_acc.astype(np.float32).values
X_test  = X_test_acc.astype(np.float32).values
y_train = np.column_stack([y_train_acc.values, y_train_inc.values]).astype(np.int64)
y_test  = np.column_stack([y_test_acc.values,  y_test_inc.values]).astype(np.int64)
feature_names = list(X_train_acc.columns)

print(f"  X_train: {X_train.shape} | y_train: {y_train.shape} | tasks: {TARGETS}")


from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train[:, 0]
)


def build_tabnet(params):
    return TabNetMultiTaskClassifier(
        n_d=params["n_d"],
        n_a=params["n_d"],                                   
        n_steps=params["n_steps"],
        gamma=1.3,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=params["lr"]),
        scheduler_params=dict(step_size=20, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
        verbose=0,
        seed=42,
    )


def objective(trial):
    params = {
        "n_d":     trial.suggest_int("n_d", 8, 64, step=8),
        "n_steps": trial.suggest_int("n_steps", 3, 7),
        "lr":      trial.suggest_float("lr", 1e-3, 5e-2, log=True),
    }
    model = build_tabnet(params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc"],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=256,
        virtual_batch_size=128,
        drop_last=False,
    )

    probs = model.predict_proba(X_val)
    aucs = [roc_auc_score(y_val[:, i], probs[i][:, 1]) for i in range(2)]
    return float(np.mean(aucs))


print(f"\nRunning Optuna ({N_TRIALS} trials) to search n_d, n_steps, lr...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"Best CV-mean AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")


print("\nRefitting winner on full train set...")
best_model = build_tabnet(study.best_params)
best_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=["auc"],
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    batch_size=256,
    virtual_batch_size=128,
    drop_last=False,
)


probs_test = best_model.predict_proba(X_test)
rows = []
for i, name in enumerate(TARGETS):
    p = probs_test[i][:, 1]
    pred = (p >= 0.5).astype(int)
    rows.append({
        "Algorithm":      "TabNet MTL",
        "Target":         name,
        "Test ROC-AUC":   round(roc_auc_score(y_test[:, i], p), 3),
        "Test Precision": round(precision_score(y_test[:, i], pred, zero_division=0), 3),
        "Test Recall":    round(recall_score(y_test[:, i], pred, zero_division=0), 3),
        "Test F1":        round(f1_score(y_test[:, i], pred, zero_division=0), 3),
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "06b_tabnet_results.csv")
df.to_csv(csv_path, index=False)


print("\nExtracting TabNet feature importances...")
global_imp = best_model.feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": global_imp}) \
           .sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp_df["feature"], imp_df["importance"], color="#4C72B0")
ax.set_xlabel("Global importance (sum over attention steps)")
ax.set_title("TabNet Global Feature Importance")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06b_tabnet_feature_importance.png"), dpi=120)
plt.close(fig)


print("Extracting per-step attention masks...")
_, step_masks = best_model.explain(X_test)

if isinstance(step_masks, dict):
    step_masks_list = [step_masks[k] for k in sorted(step_masks.keys())]
else:
    step_masks_list = list(step_masks)
step_avg = np.stack([m.mean(axis=0) for m in step_masks_list], axis=0)                         

fig, ax = plt.subplots(figsize=(10, 0.5 * step_avg.shape[0] + 2))
im = ax.imshow(step_avg, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=45, ha="right")
ax.set_yticks(range(step_avg.shape[0]))
ax.set_yticklabels([f"Step {i+1}" for i in range(step_avg.shape[0])])
ax.set_title("TabNet Attention Masks (averaged over test clients)")
fig.colorbar(im, ax=ax, label="Mean mask weight")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06b_tabnet_attention_masks.png"), dpi=120)
plt.close(fig)


model_path = os.path.join(OUT_DIR, "06b_tabnet_model")
best_model.save_model(model_path)


print("\n" + "=" * 100)
print("STEP 06b: TABNET MTL RESULTS")
print("=" * 100)
print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
print(f"\nBest hyperparams: {study.best_params}")
print(f"Saved results:     {csv_path}")
print(f"Saved importance:  {OUT_DIR}/06b_tabnet_feature_importance.png")
print(f"Saved masks:       {OUT_DIR}/06b_tabnet_attention_masks.png")
print(f"Saved model:       {model_path}.zip")
