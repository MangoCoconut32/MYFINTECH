# Deep Static Code Analysis & Architectural Audit

## 1. Hydra Schema Validation

### `DataLoader`
- Expects: `dfs_csv_path`, `raw_excel_path`, `frozen_csv_path`, `id_col`, `fold_col`, `target_cols`, `primary_target`, `n_splits`.
- Config (`configs/data/data.yaml`): Contains all these keys accurately.
- Validation: **Pass**. No `KeyError` expected.

### `FeatureEngineer`
- Expects: `base_cols`, `alois_engineered`, `dfs.enabled`, `dfs.top_n`, `corr_threshold`.
- Config (`configs/features/features.yaml`): Contains `base_cols`, `alois_engineered`, `dfs` (with `enabled`, `depth`, `top_n`, `primitives`), `corr_threshold`, and `boruta`.
- Validation: **Pass**. No `KeyError` expected.

### `XGBoostModel`
- Expects: `params` (from `hyperparameters` node) and `cfg` (with `calibration.method`, `calibration.ensemble`, `optuna.n_trials`, `search_space`).
- Config (`configs/model/xgboost.yaml`): Contains `hyperparameters` mapping identically to `XGBClassifier` inputs, `calibration`, `optuna`, `search_space`, and `artifacts`.
- Validation: **Pass**. No `KeyError` expected.

## 2. Dependency Graph & Leakage Check

### Dependency Graph Analysis
- `src/utils/logging_config.py`: Imports standard libraries (`logging`, `sys`).
- `src/models/base_model.py`: Imports standard libraries, `numpy`, `pandas`.
- `src/models/xgboost_model.py`: Imports `src.models.base_model`.
- `src/data/freezer.py`: Imports standard libraries, `numpy`, `pandas`, `sklearn`.
- `src/data/loader.py`: Imports standard libraries, `pandas`.
- `src/features/engineer.py`: Imports standard libraries, `numpy`, `pandas`.
- **Finding:** The import hierarchy is strictly linear: `models` depend on `base_model`, but no module in `data`, `features`, or `models` depends on siblings in a way that creates cycles. **No Circular Dependencies Detected.**

### Leakage Check
- **`src/data/loader.py`**: Drops `self.cfg.target_cols` in `_split_xy()`. Target columns correctly do not leak into `X` dataframes. Train/val/test data splits correctly separate targets from features.
- **`src/features/engineer.py`**: Computes medians and `p99` thresholds explicitly and exclusively on `df_train` during `fit()`, then applies those stats uniformly to `df_in` during `transform()`. The target variable is never used in feature engineering.
- **`src/models/xgboost_model.py`**: Fits on training data, validates on strictly separated CV blocks. Uses standard `predict` methods without cheating by peeking into validation targets.
- **Finding:** Target variable leakage is fully prevented. The anti-leakage protocol holds firm.

## 3. Phase 2 Preparation (Deep Learning & XAI)

### PyTorch TabNet Integration (`src/models/tabnet_model.py`)
- **Inheritance:** `TabNetModel` should inherit from `src.models.base_model.BaseFinanceModel`.
- **Config Overrides (`configs/model/tabnet.yaml`):**
  - Create a new Hydra model config using `name: "tabnet"`.
  - Include hyperparameters defined in `OLD/Main_y/04y_train_tabnet_income.py`:
    - Fixed: `n_d: 48`, `n_a: 48`, `n_steps: 3`, `gamma: 1.52`, `lambda_sparse: 0.0004`.
  - Include Optuna search spaces for `lr`, `lambda_sparse`, `gamma`, `batch_size`, `virtual_batch_size`.
- **Pre-Training (SSL):** Include logic in `fit()` or `tune()` to handle Phase 1 SSL (`TabNetPretrainer`) prior to `TabNetMultiTaskClassifier` fine-tuning.
- **Data Input View:** The original code expects a 15-feature min-max scaled view (`Train_Master_X_NN.csv`). We must introduce a `Scaler` transformer to the `FeatureEngineer` or pipeline step to support `min_max` scaling via a new config key (e.g., `features.scaling: "min_max"`), as `PipelineXTransformer` doesn't currently scale.
- **Scheduler Injection:** PyTorch `StepLR` should be passed correctly in the `fit()` implementation, matching the old script logic.

### DiCE Counterfactuals Integration (`src/xai/counterfactuals.py`)
- **Location:** Create `src/xai/__init__.py` and `src/xai/counterfactuals.py`.
- **Class Structure:** `DiCEExplainer` class taking the trained model (e.g., `XGBoostModel`) and the `DataLoader` via config.
- **Features Configuration:**
  - Define `immutable_features` (`Age`, `Gender`, `FamilyMembers`, `Age_bracket_*`) and `mutable_features` directly via a new Hydra config (`configs/xai/dice.yaml`).
- **Data Wrapper:** Wrap the `X_train` dataframe with the target column (for `dice_ml.Data`) internally, executing `generate_counterfactuals()` for a specified list of low-probability clients.
- **Output:** Save outputs to `data/processed/xai/dice_counterfactuals.csv` according to standard paths configured via Hydra `artifacts`.
