# Needs-Based Financial Recommendation System

A production-ready, end-to-end machine learning pipeline that predicts client propensity for financial investment products and delivers MIFID/IDD-compliant, explainable product recommendations.

---

## 📂 Directory Structure

```text
BuisnessCase2/
├── Dataset2_Needs.xls                    # Raw source: Needs, Products, Metadata sheets
├── README.md                             # This file
├── report.md                             # Graduate-level methodology & architectural decisions
├── TASK.md                               # SOTA backlog & experimental branch specs
│
├── Output/                               # All pipeline artifacts (git-ignored, auto-created)
│   │
│   ├── 02_baselines/
│   │   └── 02_baselines_results.csv      # Baseline sweep results (10 models × 2 targets)
│   │
│   ├── 03_grid_search/
│   │   ├── 03_grid_search_results.csv    # GridSearchCV best params & AUC scores
│   │   ├── 03_grid_search_tune_logs.txt  # Raw GridSearch trial logs
│   │   ├── 03_grid_search_{target}_rf.pkl
│   │   ├── 03_grid_search_{target}_xgb.pkl
│   │   └── 03_grid_search_{target}_svm.pkl
│   │
│   ├── 04_optuna/                        # ⭐ Central model store — consumed by 05, 07, 08
│   │   ├── 04_optuna_results.csv         # Bayesian Optuna best params & AUC scores
│   │   ├── 04_optuna_{target}_rf.pkl     # Best Optuna Random Forest (per target)
│   │   └── 04_optuna_{target}_xgb.pkl   # Best Optuna XGBoost (per target)
│   │
│   ├── 05_ensembles/
│   │   ├── 05_ensembles_results.csv      # Voting / Stacking ensemble comparisons
│   │   ├── 05_ensembles_{target}_voting_soft.pkl
│   │   ├── 05_ensembles_{target}_voting_hard.pkl
│   │   └── 05_ensembles_{target}_stacking.pkl
│   │
│   ├── 06_neural_networks/
│   │   ├── 06_neural_nets_results.csv    # MTL Neural Network final metrics
│   │   └── 06_mtl_neural_net_weights.keras  # Full Keras MTL model (arch + weights + optimizer)
│   │
│   ├── 07_XAI_Report/                   # Compliance audit artifacts (SHAP / LIME / PDP)
│   │   ├── 01_Global_SHAP_Summary.png
│   │   ├── 02_Global_Permutation_Importance.png
│   │   ├── 03_Local_SHAP_Client_10.png
│   │   ├── 04_Local_LIME_Client_10.html
│   │   └── 05_Sweep_Analysis_PDP_ICE.png
│   │
│   ├── 06b_tabnet_ssl/             # [06b experimental] TabNet SSL outputs
│   │   ├── 06b_results.csv               # AUC + F1 breakdown for both targets
│   │   ├── 06b_tabnet_ssl_model.zip      # Serialized TabNet encoder + heads
│   │   ├── 06b_Global_Feature_Importance.png
│   │   └── 06b_Local_Saliency_Map.png
│   │
│   └── 08_recommender/
│       ├── 08_recommender_coverage.csv   # Rule-by-rule coverage analysis (4 rules)
│       └── 08_client_recommendations.csv # Full per-client product recommendation table
│
└── Main_files/                           # All executable pipeline scripts
    ├── utils.py                          # Data Contract (shared across all scripts)
    ├── 01_eda.ipynb                      # Exploratory Data Analysis
    ├── 02_baselines.py                   # Classic model baseline sweep
    ├── 03_grid_search_tuning.py          # GridSearchCV hyperparameter tuning
    ├── 04_bayesian_optuna.py             # Bayesian Optuna tuning  ← run before 05, 07, 08
    ├── 05_ensembles.py                   # Voting & Stacking ensembles
    ├── 06_neural_networks.py             # SOTA Multi-Task Learning neural network (Keras)
    ├── 06b_pytorch_tabnet_ssl.py         # [06b experimental] TabNet SSL + MTL (PyTorch)
    ├── 07_xai_compliance.py              # SHAP / LIME explainability audit
    ├── 08_recommender_system.py          # Knowledge-based product recommender
    └── GPU_Runner.ipynb                  # Google Colab runner for 06b (T4 GPU)
```

> **Note on `{target}`:** Model files use the literal target name, e.g., `04_optuna_AccumulationInvestment_xgb.pkl` and `04_optuna_IncomeInvestment_xgb.pkl`.
>
> **Note on `Output/`:** This directory is git-ignored. It is created automatically when you run each script — you do not need to create it manually.


---

## 🛠️ The Data Contract (`utils.py`)

All pipeline scripts share a single data loading interface to **guarantee zero data leakage** across all phases. No script ever touches the raw Excel file directly — they all call `utils.py`.

### Key Functions

**`load_and_prepare_data(filepath, target_col, use_engineered_features=True)`**
- Loads the `Needs` sheet and isolates features from target columns.
- If `use_engineered_features=True`, appends EDA-derived transformations (`Wealth_log`, `Income_per_person`, age brackets) proven to raise ROC-AUC by ~3-4%.
- Applies a stratified `train_test_split`, then fits `MinMaxScaler` **exclusively on the training slice** — the scaler never sees test data.
- Used by: scripts `02`, `03`, `04`, `05`, `06`, `07`.

**`get_all_engineered_features(filepath)`**
- Applies the **identical** feature engineering pipeline as `load_and_prepare_data`, but returns the **complete dataset** (no train/test split, no scaling).
- Used exclusively by `08_recommender_system.py`, which needs propensity scores for **every** client rather than only a test slice.
- XGBoost is scale-invariant (tree splits are threshold-based), so MinMaxScaler is not needed for inference — only feature names must match the training schema exactly.
- ⚠️ **Maintenance rule**: any change to the feature engineering block in `load_and_prepare_data` must be mirrored here to prevent silent feature mismatch.

**`evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds=5)`**
- Universal validation wrapper compatible with any Scikit-Learn or XGBoost estimator.
- Computes 5-fold `StratifiedKFold` CV metrics, then scores independently on the hold-out test set.

---

## 🚀 Execution Order

Run scripts strictly in numeric order from inside `Main_files/`:

```bash
# Activate the virtual environment first
source ../.venv/bin/activate

python 02_baselines.py
python 03_grid_search_tuning.py
python 04_bayesian_optuna.py    # mandatory before 05, 07, 08
python 05_ensembles.py
python 06_neural_networks.py
python 07_xai_compliance.py
python 08_recommender_system.py
```

> [!IMPORTANT]
> **Scripts 05, 07, and 08 consume serialized models from Step 04** — they will exit with an error if `Output/04_optuna/` does not exist. Always run `04_bayesian_optuna.py` before any of those three scripts.

---

## 💾 Model Persistence: Saving & Loading

Every training script automatically serializes its best-performing models to `Output/` on completion. You never need to retrain from scratch.

### Model Dependency Chain

Step 04 is the **central model producer** for the pipeline:

| Downstream script | Loads from | What it uses |
|---|---|---|
| `05_ensembles.py` | `Output/04_optuna/*.pkl` | Calls `.get_params()` to build fresh unfitted instances for Voting/Stacking wrappers |
| `07_xai_compliance.py` | `Output/04_optuna/04_optuna_AccumulationInvestment_xgb.pkl` | Audits the exact deployed model — no retraining |
| `08_recommender_system.py` | `Output/04_optuna/04_optuna_*_xgb.pkl` | Calls `predict_proba` on all clients for the recommendation engine |

### Scikit-Learn / XGBoost models (`.pkl`)

Scripts `03`, `04`, and `05` use **`joblib`** to persist models:

```python
import joblib

# --- Loading a saved model ---
model = joblib.load("Output/04_optuna/04_optuna_AccumulationInvestment_xgb.pkl")

# --- Running inference ---
predictions = model.predict_proba(X_new)[:, 1]
```

### Keras Neural Network (`.keras`)

Script `06` saves the full MTL model (architecture + weights + optimizer state):

```python
import tensorflow as tf

# --- Loading the MTL network ---
model = tf.keras.models.load_model("Output/06_neural_networks/06_mtl_neural_net_weights.keras")

# --- Running dual-head inference ---
preds = model.predict(X_new)
prob_accumulation = preds[0].flatten()   # Head 1: AccumulationInvestment
prob_income       = preds[1].flatten()   # Head 2: IncomeInvestment
```

> The `.keras` format captures the complete model state — no need to rebuild the architecture before loading.

---

## 📋 Experimental Branching Convention

To try a different approach at any pipeline step **without breaking the numeric sequence or overwriting canonical results**, use an alphabetical suffix on **both the script and its outputs**.

### Rules
1. **Script name** gets the letter suffix: `02a_baselines_smote.py`, `04b_bayesian_optuna_svm.py`
2. **All `Output/` files** written by that script must also carry the same letter in their filename — this is mandatory to prevent silently overwriting the main pipeline's results.

### Example

| Canonical (main) | Experimental variant |
|---|---|
| `03_grid_search_tuning.py` | `03b_grid_search_tuning_svm_only.py` |
| `Output/03_grid_search/03_grid_search_results.csv` | `Output/03_grid_search/03b_grid_search_results.csv` |
| `Output/03_grid_search/03_grid_search_{target}_rf.pkl` | `Output/03_grid_search/03b_grid_search_{target}_rf.pkl` |

### Why this matters
- The canonical `0X_` files are the **source of truth** consumed by downstream scripts. An experimental run that overwrites them silently corrupts the entire pipeline.
- OS-level sorting keeps `03_`, `03b_`, `03c_` grouped together, making the experimental history readable at a glance.
- When a variant outperforms the canonical, **promote it**: rename the script and its outputs to the plain `0X_` form and archive the old one with the suffix.

---

## 📊 Key Results Summary

| Phase | Best Model | Target | Test ROC-AUC |
|---|---|---|---|
| 02 Baselines | XGBoost (raw) | Accumulation | ~0.83 |
| 04 Optuna | XGBoost (tuned) | Accumulation | **0.867** |
| 04 Optuna | XGBoost (tuned) | Income | 0.760 |
| 06 Neural Net | MTL Keras | Income | **0.797** |

The SOTA Multi-Task Learning network is the **production champion** for `IncomeInvestment`, while the Optuna XGBoost remains the reference model for SHAP/LIME explainability audits (Phase 07).
