"""
=============================================================================
04y_train_tabnet_income.py — TABNET V3 "PRECISION STRIKE" (NN VIEW, 15 FEATURES)
=============================================================================
PURPOSE:
    Trains TabNet SSL+MTL on the de-correlated NN Feature View (15 features).

    V3 CHANGES vs V2:
    - Input: Train_Master_X_NN.csv (30 features [0,1] scaled, 15 selected for attention)
    - Reason: NN View uses MinMaxScaler — prevents large-magnitude features
              (Wealth, Income) from dominating gradients over small-scale
              features (RiskPropensity, Gender). Tree View is NOT suitable
              for neural nets because gradient magnitude is data-magnitude-bound.
    - Precision Strike params: lr=0.0017, lambda_sparse=0.0004, gamma=1.52
      injected via enqueue_trial() as Trial 0 to guarantee they are evaluated.
    - Final Refit: max_epochs=60, no eval_set — prevents validation-triggered
      early stopping that causes 0.97+ training AUC overfit.

ARCHITECTURE:
    Phase 1 — SSL pre-training on full 4000-row Train/Val block
    Phase 2 — Optuna MTL fine-tuning with 5-fold frozen CV

INPUTS:
    Train_Master_X_NN.csv   (from 01y — 30 features [0,1] scaled + stratified_fold)
    Test_Master_X_NN.csv    (from 01y — 30 features [0,1] scaled, no fold column)

OUTPUTS (Output/Pipeline_Y/):
    04y_tabnet_income_model.zip        — TabNet weights (load with .load_model())
    04y_tabnet_pretrained.zip          — SSL pre-trainer weights
    04y_tabnet_performance.json        — AUC, Brier, P, R, F1 on blind test
    04y_tabnet_saliency_map.png        — Attention mask feature importance
=============================================================================
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score
)

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

PIPELINE_X_DIR = os.path.join(_PROJECT_ROOT, "Output", "Pipeline_Y")
TRAIN_CSV      = os.path.join(PIPELINE_X_DIR, "Train_Master_X_NN.csv")  
TEST_CSV       = os.path.join(PIPELINE_X_DIR, "Test_Master_X_NN.csv")
OUT_DIR        = PIPELINE_X_DIR
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COLS  = ["AccumulationInvestment", "IncomeInvestment"]
FOLD_COL     = "stratified_fold"
RANDOM_STATE = 42

ALOIS_15_FEATURES = [
    "Age", "Gender", "FamilyMembers", "FinancialEducation",
    "RiskPropensity", "Income", "Wealth",
    "Wealth_log", "Income_log", "Wealth_per_person", "Income_per_person",
    "Inc_to_Wealth_ratio", "Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior"
]

# Impostazioni Ottimizzate
N_TRIALS     = 30      # Aumentato per trovare i parametri perfetti
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("=" * 70)
print("04y_train_tabnet_income.py — TabNet V3 (Optimized, 15 feat)")
print("=" * 70)
print(f"      Device: {DEVICE.upper()}")

# ---------------------------------------------------------------------------
# 1. Load NN Feature View
# ---------------------------------------------------------------------------
print("\n[1/6] Loading Hybrid Feature View (15 features: 7 Base + 8 Engineered)...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

if FOLD_COL not in train_df.columns:
    print(f"❌ '{FOLD_COL}' column missing — re-run 01y.")
    sys.exit(1)

FEATURE_COLS = ALOIS_15_FEATURES
X_tv_full = train_df[FEATURE_COLS].astype(np.float32).values
y_tv_full = train_df[TARGET_COLS].astype(np.int64).values
fold_ids  = train_df[FOLD_COL].values
X_test    = test_df[FEATURE_COLS].astype(np.float32).values
y_test_np = test_df[TARGET_COLS].astype(np.int64).values

print(f"✅ Data Contract Verified: Loaded {X_tv_full.shape[1]} features")

# ---------------------------------------------------------------------------
# 2. Phase 1 — Self-Supervised Pre-Training (1 esecuzione fissa)
# ---------------------------------------------------------------------------
print("\n[2/6] Phase 1 — SSL Pre-training on full 4000 rows...")

# Dimensioni Architetturali fisse per garantire la compatibilità
N_D = 48
N_A = 48
N_STEPS = 3

pretrainer = TabNetPretrainer(
    n_d=N_D, n_a=N_A, n_steps=N_STEPS,
    gamma=1.3,
    momentum=0.02,
    mask_type="entmax",
    seed=RANDOM_STATE,
    device_name=DEVICE,
    verbose=0,
)

pretrainer.fit(
    X_train=X_tv_full,
    eval_set=[X_tv_full],
    pretraining_ratio=0.20,
    max_epochs=200,
    patience=30,
    batch_size=256,
    virtual_batch_size=128,
    drop_last=False,
    num_workers=0,
)

pretrain_path = os.path.join(OUT_DIR, "04y_tabnet_pretrained")
pretrainer.save_model(pretrain_path)

# ---------------------------------------------------------------------------
# 3. Phase 2 — Optuna Optimization (Ricerca dei parametri comportamentali)
# ---------------------------------------------------------------------------
print(f"\n[3/6] Phase 2 — Optuna ({N_TRIALS} trials × 5 folds)...")

def objective(trial):
    # Ricerca parametri ottimizzata
    lr            = trial.suggest_float("lr",           1e-4, 3e-2, log=True)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True)
    gamma         = trial.suggest_float("gamma",         1.0, 2.0)
    weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    batch_size    = trial.suggest_categorical("batch_size", [256, 512])
    v_batch_choices = [64, 128, 256]
    valid_v_batches = [v for v in v_batch_choices if v <= batch_size]
    virtual_batch_size = trial.suggest_categorical("virtual_batch_size", valid_v_batches)

    fold_aucs = []
    for fold_id in range(5):
        val_mask   = fold_ids == fold_id
        train_mask = ~val_mask

        X_tr, y_tr = X_tv_full[train_mask], y_tv_full[train_mask]
        X_vl, y_vl = X_tv_full[val_mask], y_tv_full[val_mask]

        clf = TabNetMultiTaskClassifier(
            n_d=N_D, n_a=N_A, n_steps=N_STEPS,
            gamma=gamma, lambda_sparse=lambda_sparse, momentum=0.02,
            mask_type="entmax", optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": lr, "weight_decay": weight_decay},
            
            # CORREZIONE APPLICATA: StepLR compatibile con TabNet
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 20, "gamma": 0.5},
            
            seed=RANDOM_STATE, device_name=DEVICE, verbose=0,
        )

        clf.fit(
            X_train=X_tr, y_train=y_tr,
            eval_set=[(X_vl, y_vl)], eval_name=["val"], eval_metric=["auc"],
            max_epochs=200, patience=25, batch_size=batch_size,
            virtual_batch_size=virtual_batch_size, drop_last=False,
            num_workers=0, from_unsupervised=pretrainer,
        )

        probs = clf.predict_proba(X_vl)
        auc_inc = roc_auc_score(y_vl[:, 1], probs[1][:, 1])
        fold_aucs.append(auc_inc)

    return float(np.mean(fold_aucs))

# Precision Strike (Manteniamo i parametri promettenti come punto di partenza)
v3_best_params = {'lr': 0.0017, 'lambda_sparse': 0.0004, 'gamma': 1.52, 'batch_size': 256, 'virtual_batch_size': 128, 'weight_decay': 1e-4}

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.enqueue_trial(v3_best_params)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
print(f"\n      Best 5-Fold Inc AUC : {study.best_value:.4f}")
print(f"      Best params         : {best_params}")

# ---------------------------------------------------------------------------
# 4. Final Refit (Anti-Overfitting Mode)
# ---------------------------------------------------------------------------
print("\n[4/6] Final refit on full Train/Val block with best params...")

clf_final = TabNetMultiTaskClassifier(
    n_d=N_D, n_a=N_A, n_steps=N_STEPS,
    gamma=best_params["gamma"], lambda_sparse=best_params["lambda_sparse"],
    momentum=0.02, mask_type="entmax", optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": best_params["lr"], "weight_decay": best_params.get("weight_decay", 1e-4)},
    
    # Usiamo StepLR in fase finale perché usiamo tutti i dati e non abbiamo un eval_set da monitorare
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"step_size": 30, "gamma": 0.5},
    
    seed=RANDOM_STATE, device_name=DEVICE, verbose=1,
)

clf_final.fit(
    X_train=X_tv_full, y_train=y_tv_full,
    max_epochs=55, # LIMITE RIGIDO per evitare overfitting senza validation set
    patience=10, 
    batch_size=best_params["batch_size"],
    virtual_batch_size=best_params.get("virtual_batch_size", 128),
    drop_last=False, num_workers=0, from_unsupervised=pretrainer,
)

model_path = os.path.join(OUT_DIR, "04y_tabnet_income_model")
clf_final.save_model(model_path)

# ---------------------------------------------------------------------------
# 5. Blind Test Evaluation
# ---------------------------------------------------------------------------
print("\n[5/6] Evaluating on blind Test Set...")

probs_test = clf_final.predict_proba(X_test)
p_acc = probs_test[0][:, 1]
p_inc = probs_test[1][:, 1]

y_acc_true = y_test_np[:, 0]
y_inc_true = y_test_np[:, 1]

results = {}
for target, y_true, p_pred in [
    ("AccumulationInvestment", y_acc_true, p_acc),
    ("IncomeInvestment",       y_inc_true, p_inc),
]:
    pred = (p_pred >= 0.5).astype(int)
    results[target] = {
        "AUC": round(roc_auc_score(y_true, p_pred), 4),
        "Brier": round(brier_score_loss(y_true, p_pred), 4),
        "Precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, pred, zero_division=0), 4),
    }
    print(f"\n      {target} AUC: {results[target]['AUC']:.4f}")

perf_path = os.path.join(OUT_DIR, "04y_tabnet_performance.json")
with open(perf_path, "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------------------
# 6. Saliency Map & Learning Curves
# ---------------------------------------------------------------------------
print("\n[6/6] Generating attention saliency map...")
try:
    explain_output = clf_final.explain(X_test)
    M_explain = explain_output[0] 
    
    if isinstance(M_explain, np.ndarray) and M_explain.ndim == 2:
        mean_attention = M_explain.mean(axis=0)
    elif isinstance(M_explain, list):
        mean_attention = np.stack([m.mean(axis=0) for m in M_explain]).mean(axis=0)
    else:
        raise ValueError("Unexpected explain() output type")

    mean_attention = mean_attention / (mean_attention.sum() + 1e-9)
    imp_series = pd.Series(mean_attention, index=FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#1B3A6B" if val >= imp_series.median() else "#A9CCE3" for val in imp_series.values]
    ax.barh(imp_series.index, imp_series.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("TabNet V3 — Feature Attention Saliency Map", fontsize=16, fontweight="bold", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    
    fig.tight_layout()
    saliency_path = os.path.join(OUT_DIR, "04y_tabnet_saliency_map.png")
    fig.savefig(saliency_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
except Exception as e:
    print(f"      ⚠️  Saliency map failed ({e})")

print("\n[6b/6] Generating learning curves...")
try:
    if hasattr(clf_final, 'history') and 'loss' in clf_final.history and len(clf_final.history['loss']) > 0:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        epochs = range(len(clf_final.history['loss']))
        
        ax1.plot(epochs, clf_final.history['loss'], color='#1B3A6B', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss (LogLoss)', fontsize=12, color='#1B3A6B')
        
        ax1.set_title("TabNet V3 — Training Convergence (Phase 2 Refit)", fontsize=15, fontweight='bold', loc='left')
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.spines['top'].set_visible(False)
        
        fig.tight_layout()
        curve_path = os.path.join(OUT_DIR, "04y_tabnet_learning_curves.png")
        fig.savefig(curve_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("      Saved: 04y_tabnet_learning_curves.png")
    else:
        print("      ℹ️  Learning history not available for this refit (skipping plot).")
except Exception as e:
    print(f"      ⚠️  Learning curves failed ({e})")

print("\n" + "=" * 70)
print("✅ 04y_train_tabnet_income.py (V3 Optimized) COMPLETE")
print("=" * 70)