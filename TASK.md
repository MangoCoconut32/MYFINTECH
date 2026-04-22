# 🚀 SOTA Optimization Backlog: Expanded Dev Specifications

**Objective:** Push the current production pipeline to State-of-the-Art (SOTA) using isolated experimental branches (`_b`, `_c` variants).

---

## 🟢 02: Baselines & Algorithm Selection

### [ ] `02b_catboost_lightgbm.py` (Modern GBDTs)
**Dev Brief:** * **Libraries:** `catboost`, `lightgbm`.
* **Instruction:** Replace standard Random Forest/XGBoost baselines. For CatBoost, you MUST utilize its native categorical handling. Do not One-Hot Encode categorical variables for CatBoost; instead, pass them directly using the `cat_features` parameter. For LightGBM, ensure `class_weight='balanced'` is tested.
* **Output:** A new `02b_catboost_lgbm_results.csv` comparing their Test ROC-AUC against our baseline XGBoost.

### [ ] `02c_tabpfn_baseline.py` (Zero-Shot Deep Learning)
**Dev Brief:**
* **Libraries:** `tabpfn`.
* **Instruction:** Import `TabPFNClassifier`. Do NOT perform any hyperparameter tuning (no Optuna/GridSearchCV). Do NOT normalize the data (TabPFN handles this internally). Fit it directly on the train set and predict on the test set. 
* **Note to Dev:** TabPFN scales quadratically. Ensure the dataset is under 10,000 rows (our ~4000 is perfect) and limit inference batch sizes if memory errors occur.

---

## 🟡 03: Feature Engineering & Selection

### [ ] `03b_boruta_shap.py` (Adversarial Feature Selection)
**Dev Brief:**
* **Libraries:** `BorutaShap` (pip install BorutaShap).
* **Instruction:** Instantiate `BorutaShap` wrapping our best Phase 2 XGBoost model. Run `.fit(X_train, y_train, n_trials=50, random_state=42)`. 
* **Output:** The dev MUST automatically extract the "Accepted" features, print the "Rejected" (noise) features, and save a `03b_boruta_selected_features.json` file. Future pipelines will read this JSON to drop useless columns before training.

### [ ] `03c_deep_feature_synthesis.py` (Automated Feature Engineering)
**Dev Brief:**
* **Libraries:** `featuretools`, `lightgbm`.
* **Instruction:** Create a Featuretools `EntitySet` from our dataframe. Run `ft.dfs(target_entity="clients", max_depth=2)` using primitives like `['add', 'multiply', 'divide', 'percentile']`. 
* **Constraint:** DFS will generate hundreds of columns. The dev MUST instantly train a LightGBM model on this massive matrix, extract `feature_importances_`, and keep ONLY the Top 30 features to prevent the curse of dimensionality. Save the new dataset as `Dataset2_Needs_DFS.csv`.

---

## 🟠 04: Hyperparameter Optimization

### [ ] `04b_optuna_multiobjective.py` (AUC + Calibration)
**Dev Brief:**
* **Libraries:** `optuna`, `sklearn.metrics.brier_score_loss`.
* **Instruction:** Rewrite the Optuna objective function to return a tuple of TWO metrics: `(roc_auc, brier_score)`. Update the study to `directions=["maximize", "minimize"]`.
* **Constraint:** Because this creates a Pareto front instead of a single best model, `study.best_params` will fail. The dev MUST write a custom selection function: *Find the trial in `study.best_trials` that has the highest ROC-AUC strictly subject to the constraint that Brier Score < 0.15*. 

---

## 🔴 05: Ensembles

### [ ] `05b_explainable_boosting_machines.py` (Glassbox Ensembles)
**Dev Brief:**
* **Libraries:** `interpret` (Microsoft InterpretML).
* **Instruction:** Train an `ExplainableBoostingClassifier` (EBM). EBMs are generalized additive models (GAMs) with pairwise interactions. 
* **Output:** Compare its AUC to the Optuna-XGBoost. Then, export the built-in global explanation dashboard (`ebm.explain_global()`) to an HTML file in the Output folder. Show how this bypasses the need for the heavy SHAP library entirely.

---

## 🟣 06: Neural Networks

### [ ] `06b_tabnet_mtl.py` (Attention-Based Tabular DL)
**Dev Brief:**
* **Libraries:** `pytorch-tabnet`.
* **Instruction:** Import `TabNetMultiTaskClassifier`. This is the SOTA replacement for our Keras MLP. 
* **Parameters to tune:** Ask Optuna to tune `n_d` and `n_a` (between 8 and 64, keeping `n_d == n_a`), and `n_steps` (between 3 and 7). 
* **Output:** Aside from metrics, the dev MUST extract the `feature_importances_` masks from the trained TabNet and save them as a heatmap plot to show *where* the neural network was "looking" when making its decisions.

---

## 🔵 07: Explainable AI & Compliance

### [ ] `07b_counterfactual_explanations.py` (DiCE)
**Dev Brief:**
* **Libraries:** `dice_ml`.
* **Instruction:** Initialize a DiCE explainer (`dice_ml.Model`, `dice_ml.Data`, `dice_ml.Dice`). Pick 5 clients who were rejected for "IncomeInvestment". 
* **Task:** Generate 3 counterfactuals for each client showing the *minimal mathematical change* required to get them approved. 
* **Constraint:** The dev MUST lock immutable features. For example, Age cannot decrease, and Family Members cannot be fractional. Only allow `Wealth`, `Income`, and `RiskPropensity` to vary in the counterfactual generation.

---

## 🟤 08: Recommender System

### [ ] `08b_two_tower_embeddings.py` (Neural Vector Matching)
**Dev Brief:**
* **Libraries:** `tensorflow` or `pytorch`.
* **Instruction:** Build a Two-Tower architecture. 
  * Tower A (Client Encoder): Dense layers reducing client features to a 64-dimensional embedding vector.
  * Tower B (Product Encoder): Dense layers reducing product metadata (Risk, Cost, Category) to a 64-dimensional embedding vector.
* **Mechanism:** The recommendation score is the Dot Product (cosine similarity) between the Client Vector and the Product Vector. 
* **Output:** Train using a contrastive loss (pulling matches together, pushing mismatches apart). Output a `08b_neural_recommendations.csv` mapping the top 3 highest dot-product products for each client.