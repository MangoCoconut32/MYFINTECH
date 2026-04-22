"""
=============================================================================
STEP 06c - SELF-SUPERVISED TABNET (PYTORCH) — MULTI-TASK LEARNING
=============================================================================
PURPOSE:
    This is the SOTA experimental variant of Step 06, targeting a breakthrough
    on the IncomeInvestment AUC ceiling (currently 0.797 with the Keras MTL).

    The approach combines two powerful ideas:

    PHASE 1 — Self-Supervised Pre-training (SSL):
        TabNetPretrainer learns rich feature representations WITHOUT labels by
        masking a fraction of the input features and training the network to
        reconstruct them from the unmasked subset. On a small dataset (~5k
        rows), this is a crucial regularization technique: the model learns the
        joint distribution of features before ever seeing a label, providing
        the fine-tuner with a meaningful initialization rather than random noise.

    PHASE 2 — Multi-Task Fine-Tuning:
        TabNetMultiTaskClassifier is warm-started from the pretrained encoder
        (via `from_unsupervised`) and simultaneously optimizes for both targets:
          - AccumulationInvestment (T1, easier — XGBoost baseline: 0.867 AUC)
          - IncomeInvestment       (T2, harder — primary SOTA target)

        The key advantage of TabNet over our Keras trunk is the attention
        mechanism: at each of the N decision steps, a sparse attention mask
        selects WHICH features to use. This is instance-specific — different
        clients may trigger attention on different features — which makes TabNet
        naturally interpretable and better suited for heterogeneous client data.

WHY NOT JUST USE 06_neural_networks.py?
    The Keras MLP applies all features uniformly at every layer. Tab-Net
    sequentially selects features via sparse attention, which forces feature
    selection and produces exact (not approximate) feature importance maps
    without a separate SHAP call. It is the SOTA architecture for tabular
    classification as of Arik & Pfister (2021).

BRANCHING CONVENTION:
    This is a standalone experimental script (suffix 'b'). It does NOT
    overwrite any canonical pipeline file, and ALL its outputs are prefixed
    with '06c_'. Following the pipeline rule: the output subfolder is named
    after the script, i.e. Output/06c_tabnet_ssl/.

INPUTS:
    - Dataset2_Needs.xls    (Needs sheet — same as all pipeline scripts)
    - utils.py              (get_all_engineered_features for feature consistency)

OUTPUTS (all in Output/06c_tabnet_ssl/):
    - 06c_results.csv                    AUC + F1 breakdown for both targets
    - 06c_tabnet_ssl_model.zip           Serialized TabNet encoder + heads
    - 06c_Global_Feature_Importance.png  Bar chart: mean attention per feature
    - 06c_Local_Saliency_Map.png         Heatmap: 50 clients × features

TO RELOAD THE SAVED MODEL ON INFERENCE:
    from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
    clf = TabNetMultiTaskClassifier()
    clf.load_model("Output/06c_tabnet_ssl/06c_tabnet_ssl_model.zip")
    preds_proba = clf.predict_proba(X_new_np)  # list[array(n,2), array(n,2)]
    prob_acc = preds_proba[0][:, 1]
    prob_inc = preds_proba[1][:, 1]

GPU NOTE:
    This script is designed for headless execution on Google Colab (T4 GPU).
    Use the companion GPU_Runner.ipynb to execute it in the cloud.
    On CPU it will still run correctly but significantly slower.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import matplotlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabnet.*")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
print("🔒 PyTorch Deterministic Seed locked (42).")

matplotlib.use('Agg')   # Non-interactive backend — safe for Colab / headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_tabnet.*")

# pytorch-tabnet — install: pip install pytorch-tabnet
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------
# os.path.abspath(__file__) is robust regardless of the caller's working
# directory — this mirrors the convention used in all other pipeline scripts.
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(script_dir, "Dataset2_Needs.xls")

# utils.py lives in the same directory — insert so Python can find it
sys.path.insert(0, script_dir)
import optuna
from utils import load_and_prepare_data

if not os.path.exists(FILE_PATH):
    print("ERROR: Could not find Dataset2_Needs.xls. Check FILE_PATH.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Output Directory
# ---------------------------------------------------------------------------
# Default: one level up from Main_files/ → BuisnessCase2/Output/06c_tabnet_ssl/
# Override: set env var TABNET_SSL_OUTPUT_DIR (e.g. from the Colab notebook)
# so that on Colab the outputs always land inside the Fintech project folder
# regardless of where the script file itself is placed on Drive.
_env_out = os.environ.get("TABNET_SSL_OUTPUT_DIR")
output_dir = _env_out if _env_out else os.path.normpath(
    os.path.join(script_dir, "..", "Output", "06c_tabnet_ssl")
)
os.makedirs(output_dir, exist_ok=True)

print("=" * 100)
print("STEP 06c: SELF-SUPERVISED TABNET (PYTORCH) — MULTI-TASK LEARNING")
print("=" * 100)
print(f" -> Device : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
print(f" -> Output : {output_dir}")

TARGET_NAMES = ["AccumulationInvestment", "IncomeInvestment"]
N_TRIALS = 15

# ---------------------------------------------------------------------------
# 1. Data Loading & Preparation (Strict Contract)
# ---------------------------------------------------------------------------
print("\n[1] Loading and preparing data...")

X_train_acc, X_test_acc, y_train_acc, y_test_acc = load_and_prepare_data(
    FILE_PATH, "AccumulationInvestment", use_engineered_features=True
)

full_needs = pd.read_excel(FILE_PATH, sheet_name="Needs")
full_needs.columns = full_needs.columns.str.strip()

y_train_inc = full_needs.loc[X_train_acc.index, "IncomeInvestment"]
y_test_inc  = full_needs.loc[X_test_acc.index,  "IncomeInvestment"]

feature_names = list(X_train_acc.columns)

X_train = X_train_acc.astype(np.float32).values
X_test  = X_test_acc.astype(np.float32).values

y_train = np.column_stack([y_train_acc.values, y_train_inc.values]).astype(np.int64)
y_test  = np.column_stack([y_test_acc.values,  y_test_inc.values]).astype(np.int64)

print(f" -> Train : {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f" -> Features: {X_train.shape[1]} | Tasks: {y_train.shape[1]}")

# ---------------------------------------------------------------------------
# 2. Phase 1 & 2 Wrapped in Optuna Objective (3-Fold CV)
# ---------------------------------------------------------------------------
def objective(trial):
    n_d = trial.suggest_int("n_d", 16, 64, step=16)
    n_a = n_d
    n_steps = trial.suggest_int("n_steps", 3, 7)
    lr = trial.suggest_float("lr", 1e-3, 5e-2, log=True)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 1.0, 2.0)
    
    TABNET_SHARED_PARAMS = dict(
        n_d=n_d, n_a=n_a, n_steps=n_steps,
        gamma=gamma, lambda_sparse=lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": lr, "weight_decay": 1e-5},
        mask_type="sparsemax",
        n_shared=2, n_independent=2,
        verbose=0,
        seed=42
    )

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_aucs = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr_fold = X_train[train_idx]
        y_tr_fold = y_train[train_idx]
        X_va_fold = X_train[val_idx]
        y_va_fold = y_train[val_idx]
        
        pretrainer = TabNetPretrainer(**TABNET_SHARED_PARAMS)
        pretrainer.fit(
            X_train=X_tr_fold, eval_set=[X_va_fold], pretraining_ratio=0.2,
            max_epochs=70, patience=15, batch_size=256, virtual_batch_size=32, drop_last=False
        )

        # Calculation of balanced sample weights for the Income target (index 1)
        sample_weights = compute_sample_weight('balanced', y_tr_fold[:, 1])

        clf = TabNetMultiTaskClassifier(**TABNET_SHARED_PARAMS)
        clf.fit(
            X_train=X_tr_fold, y_train=y_tr_fold, eval_set=[(X_va_fold, y_va_fold)],
            eval_metric=["auc"], from_unsupervised=pretrainer,
            max_epochs=70, patience=15, batch_size=256, virtual_batch_size=32, 
            weights=sample_weights, drop_last=False
        )

        probs = clf.predict_proba(X_va_fold)
        # Target 1 is IncomeInvestment
        fold_income_auc = roc_auc_score(y_va_fold[:, 1], probs[1][:, 1])
        fold_aucs.append(fold_income_auc)
    
    mean_auc = float(np.mean(fold_aucs))
    print(f"\n🔥 Trial {trial.number+1}/{N_TRIALS} Completed | Income AUC (3-Fold CV): {mean_auc:.4f}")
    print(f"   Params: n_d={n_d}, n_steps={n_steps}, lr={lr:.5f}, gamma={gamma:.2f}")
    print("-" * 60)
    return mean_auc

print(f"\n[2] Running Optuna ({N_TRIALS} trials)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f" -> Best Valid Mean AUC: {study.best_value:.4f}")
print(f" -> Best Params: {study.best_params}")

# ---------------------------------------------------------------------------
# 3. Refit Winner
# ---------------------------------------------------------------------------
print("\n[3] Refitting best setup on full X_train...")
best_params = dict(
    n_d=study.best_params["n_d"], n_a=study.best_params["n_d"],
    n_steps=study.best_params["n_steps"],
    gamma=study.best_params["gamma"],                  # <--- FIX: Dinamico
    lambda_sparse=study.best_params["lambda_sparse"],  # <--- FIX: Dinamico
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": study.best_params["lr"], "weight_decay": 1e-5},
    mask_type="sparsemax",
    n_shared=2, n_independent=2,
    verbose=1,
    seed=42
)

final_pretrainer = TabNetPretrainer(**best_params)
final_pretrainer.fit(
    X_train=X_train, eval_set=[X_test], pretraining_ratio=0.2,
    max_epochs=150, patience=20, batch_size=256, virtual_batch_size=64, drop_last=False
)

# Calculate weights for the entire training set based on Income
final_weights = compute_sample_weight('balanced', y_train[:, 1])

clf = TabNetMultiTaskClassifier(**best_params)
clf.fit(
    X_train=X_train, y_train=y_train, eval_set=[(X_test, y_test)],
    eval_metric=["auc"], from_unsupervised=final_pretrainer,
    max_epochs=250, patience=25, batch_size=256, virtual_batch_size=64, 
    weights=final_weights, drop_last=False
)

# ---------------------------------------------------------------------------
# 5. Model Persistence
# ---------------------------------------------------------------------------
# TabNet saves as a .zip archive containing:
#   - network architecture (JSON)
#   - weights (PyTorch state_dict)
# The path argument should NOT include the .zip extension — TabNet adds it.
model_path = os.path.join(output_dir, "06c_tabnet_ssl_model")
clf.save_model(model_path)
print(f"\n -> Model saved to: {model_path}.zip")


# ---------------------------------------------------------------------------
# 6. Evaluation on Hold-Out Test Set
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[4] EVALUATION — Hold-Out Test Set")
print("=" * 80)

# predict_proba returns a Python list of length n_tasks.
# Each element is a numpy array of shape (n_test, n_classes).
# Column index 1 = P(class=1) = propensity score.
preds_proba = clf.predict_proba(X_test)

results_rows = []
for i, target in enumerate(TARGET_NAMES):
    proba_pos  = preds_proba[i][:, 1]          # probability of positive class
    y_true     = y_test[:, i]
    pred_class = (proba_pos >= 0.5).astype(int)

    auc    = roc_auc_score(y_true, proba_pos)
    report = classification_report(y_true, pred_class, output_dict=True, zero_division=0)

    f1_macro   = report["macro avg"]["f1-score"]
    f1_class1  = report["1"]["f1-score"]
    prec_class1= report["1"]["precision"]
    rec_class1 = report["1"]["recall"]

    results_rows.append({
        "Algorithm":            "TabNet SSL + MTL (PyTorch)",
        "Target":               target,
        "Test ROC-AUC":         round(auc, 4),
        "F1 Macro":             round(f1_macro, 4),
        "F1 Class-1":           round(f1_class1, 4),
        "Precision Class-1":    round(prec_class1, 4),
        "Recall Class-1":       round(rec_class1, 4),
    })

    print(f"\n{'─'*60}")
    print(f"TARGET: {target}")
    print(f"{'─'*60}")
    print(f"  Test ROC-AUC : {auc:.4f}")
    print(f"  Classification Report:\n")
    print(classification_report(y_true, pred_class, zero_division=0))


df_results = pd.DataFrame(results_rows)

print("\n" + "=" * 120)
print("STEP 06c: TABNET SSL RESULTS SUMMARY")
print("=" * 120)
print(tabulate(df_results, headers="keys", tablefmt="grid", showindex=False))

csv_path = os.path.join(output_dir, "06c_results.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n -> Results saved to: {csv_path}")


# ---------------------------------------------------------------------------
# 7. Explainability — Feature Importance & Saliency Map
# ---------------------------------------------------------------------------
# clf.explain(X) is an EXACT operation — not an approximation like SHAP.
# It returns:
#   explain_matrix : (n_samples, n_features)
#     The aggregated attention weight each feature received across ALL decision
#     steps for each sample. High value = this feature was heavily consulted.
#
#   masks : dict { step_index: (n_samples, n_features) }
#     The raw step-level attention masks before aggregation. Useful for
#     understanding WHICH step attended to which features.
#
# This eliminates the need for a separate SHAP call for this model variant.
print("\n[5] Generating explainability visualizations...")

explain_matrix, masks = clf.explain(X_test)
# explain_matrix shape: (n_test_samples, n_features)

# ── A. Global Feature Importance ──────────────────────────────────────────
# Average attention weight across all test clients → one scalar per feature.
# Features with high mean attention are the ones TabNet consistently
# "chose to look at" regardless of individual client characteristics.
global_importance = explain_matrix.mean(axis=0)      # shape: (n_features,)
mean_importance   = global_importance.mean()

# Sort descending so the most important feature is at the top of the chart
sorted_idx    = np.argsort(global_importance)[::-1]
sorted_names  = [feature_names[i] for i in sorted_idx]
sorted_vals   = global_importance[sorted_idx]

# Colour gradient: viridis from lower-bound 0.3 to 0.9 for visual clarity
colours = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_names)))

fig, ax = plt.subplots(figsize=(13, 6))
ax.barh(
    sorted_names[::-1], sorted_vals[::-1],   # flip so max is at top
    color=colours, edgecolor="white", linewidth=0.4
)
ax.axvline(
    mean_importance, color="crimson", linestyle="--", linewidth=1.4,
    label=f"Mean = {mean_importance:.4f}"
)
ax.set_xlabel("Mean Attention Weight (aggregated across all decision steps)", fontsize=12)
ax.set_title(
    "TabNet Global Feature Importance\n"
    "(Exact attention-based — no SHAP approximation)",
    fontsize=14, fontweight="bold", pad=14
)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

gi_path = os.path.join(output_dir, "06c_Global_Feature_Importance.png")
fig.savefig(gi_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f" -> Global Feature Importance saved: {gi_path}")


# ── B. Local Saliency Map (heatmap) ───────────────────────────────────────
# Show attention weights for the first N_CLIENTS test-set clients.
# Row   = one client (instance-level saliency)
# Column = one feature
# Colour = attention weight — how much the model "looked" at this feature
#          when making its decision for this specific client.
#
# Features are ordered by global importance (left = most important) so
# structure in the heatmap is easier to read.
#
# A domain-expert can use this heatmap to identify:
#   - which features drive decisions for specific client segments
#   - anomalous clients where TabNet attends to unexpected features
#     (potential data quality issue or interesting edge case)

N_CLIENTS          = min(50, len(X_test))
saliency_data      = explain_matrix[:N_CLIENTS]               # (50, n_features)
saliency_sorted    = saliency_data[:, sorted_idx]             # reorder by importance
col_labels_sorted  = sorted_names

fig, ax = plt.subplots(figsize=(18, 11))
sns.heatmap(
    saliency_sorted,
    ax          = ax,
    cmap        = "YlOrRd",
    xticklabels = col_labels_sorted,
    yticklabels = [f"Client {i+1}" for i in range(N_CLIENTS)],
    linewidths  = 0.25,
    linecolor   = "white",
    cbar_kws    = {"label": "Attention Weight", "shrink": 0.55, "pad": 0.02}
)
ax.set_title(
    f"TabNet Local Saliency Map — First {N_CLIENTS} Test Clients\n"
    "(Features ordered by global importance ▶ most attended on left)",
    fontsize=14, fontweight="bold", pad=16
)
ax.set_xlabel("Feature (sorted by global importance)", fontsize=11)
ax.set_ylabel(f"Client (test set, n={N_CLIENTS})", fontsize=11)
plt.xticks(rotation=40, ha="right", fontsize=9)
plt.yticks(fontsize=8)
fig.tight_layout()

sm_path = os.path.join(output_dir, "06c_Local_Saliency_Map.png")
fig.savefig(sm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f" -> Local Saliency Map saved:         {sm_path}")

# ---------------------------------------------------------------------------
# 8. Final Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("STEP 06c COMPLETE — All outputs written to:")
print(f"  {output_dir}")
print("=" * 100)
print("\nFiles generated:")
for f in sorted(os.listdir(output_dir)):
    fpath = os.path.join(output_dir, f)
    size  = os.path.getsize(fpath)
    print(f"  {f:<50}  {size / 1024:>8.1f} KB")
