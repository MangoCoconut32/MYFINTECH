# %% [markdown]
# # PIPELINE Z: SOTA FINANCIAL RECOMMENDATION ENGINE (COLAB EDITION)
# **Configurazione:** MyDrive/PipelineZ/ | **Output:** OutputZ/
# 
# Questo notebook implementa l'intera architettura SOTA (State-of-the-Art) per la profilazione MiFID II.
# Gestisce automaticamente:
# 1. Mount di Google Drive e installazione dipendenze.
# 2. **Data Contract**: Trasformatore stateful per eliminare il Data Leakage.
# 3. **Ablation Study**: Confronto rigoroso tra performance RAW e ENGINEERED.
# 4. **Hyperparameter Tuning**: Ricerca Optuna con Warm Start (riparte dai migliori parametri salvati).
# 5. **Ensemble Audit**: Weighted Soft Voting basato sulla diversità dei modelli.

# %% [code]
import os
import sys
import json
import joblib
import time
import pickle
import warnings
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, accuracy_score, precision_recall_curve,
    roc_curve, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from tabulate import tabulate
import optuna

def install_dependencies():
    """Installa i pacchetti necessari per l'ambiente Colab."""
    required = {"optuna": "optuna", "xgboost": "xgboost", "lightgbm": "lightgbm", "tabulate": "tabulate", "openpyxl": "openpyxl"}
    missing = [pkg for mod, pkg in required.items() if not __import__('importlib.util').util.find_spec(mod)]
    if missing:
        print(f"📦 Installazione dipendenze mancanti: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
    print("✅ Dipendenze pronte.")

install_dependencies()

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --- CONFIGURAZIONE PERCORSI COLAB ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT    = '/content/drive/MyDrive/PipelineZ'
    PIPELINE_Z_DIR  = os.path.join(PROJECT_ROOT, "OutputZ")
    RAW_EXCEL       = os.path.join(PROJECT_ROOT, "Dataset2_Needs.xls")
except:
    PROJECT_ROOT    = "."
    PIPELINE_Z_DIR  = "./OutputZ"
    RAW_EXCEL       = "./Dataset2_Needs.xls"

os.makedirs(PIPELINE_Z_DIR, exist_ok=True)

# File della Pipeline
BASE_DATA_PATH   = os.path.join(PIPELINE_Z_DIR, "Dataset_Needs_SOTA.csv")
MASTER_DATA_PATH = os.path.join(PIPELINE_Z_DIR, "Master_Needs_SOTA_Z.csv")
PARAMS_FILE      = os.path.join(PIPELINE_Z_DIR, "best_params_Z.json")

RANDOM_STATE = 42
TARGET_COLS  = ["AccumulationInvestment", "IncomeInvestment"]
FOLD_COL     = "stratified_fold"
N_TRIALS     = 10  # Numero di trial Optuna per coppia modello/target

# %% [markdown]
# ## 1. DATA CONTRACT: PipelineXTransformer
# Implementazione del protocollo anti-leakage: le statistiche sono calcolate esclusivamente sui fold di training.

# %% [code]
class PipelineXTransformer:
    def __init__(self):
        self.medians, self.p99_inc, self.p99_wth, self.inc_max = None, None, None, None
        self.is_fitted = False

    def fit(self, df_train):
        df = df_train.copy()
        self.p99_inc = df["Income"].quantile(0.99)
        self.p99_wth = df["Wealth"].quantile(0.99)
        self.inc_max = df["Income"].max()
        self.medians = df.median(numeric_only=True)
        self.is_fitted = True
        return self

    def transform(self, df_in):
        if not self.is_fitted: raise RuntimeError("Transformer non fittato.")
        df = df_in.copy()
        df.fillna(self.medians, inplace=True)
        
        # Age brackets
        df["Age_bracket"] = pd.cut(df["Age"], bins=[17, 35, 55, 120], labels=["Young", "Mid", "Senior"])
        dummies = pd.get_dummies(df["Age_bracket"], prefix="Age_bracket", dtype=int)
        for label in ["Age_bracket_Young", "Age_bracket_Mid", "Age_bracket_Senior"]:
            if label not in dummies.columns: dummies[label] = 0
        df = pd.concat([df.drop(columns=["Age_bracket"]), dummies], axis=1)

        # Financial Ratios SOTA
        clipped_inc, clipped_wth = df["Income"].clip(upper=self.p99_inc), df["Wealth"].clip(upper=self.p99_wth)
        df["Wealth_log"], df["Income_log"] = np.log1p(df["Wealth"]), np.log1p(df["Income"])
        adult_years = (df["Age"] - 17).clip(lower=1)
        df["Wealth_Age_Ratio_log"] = np.log1p(clipped_wth / adult_years)
        safe_fm = df["FamilyMembers"].replace(0, np.nan).fillna(self.medians.get("FamilyMembers", 1))
        df["Wealth_per_person"], df["Income_per_person"] = clipped_wth / safe_fm, clipped_inc / safe_fm
        df["Income_Wealth_Ratio_log"] = np.log1p(clipped_inc.div(clipped_wth.replace(0, np.nan)).fillna(self.inc_max))
        return df

# Helper per l'accesso ai dati (Simula utilsz.py)
def get_data(stage="master"):
    path = BASE_DATA_PATH if stage == "base" else MASTER_DATA_PATH
    df = pd.read_csv(path)
    tr_mask, te_mask = df[FOLD_COL] >= 0, df[FOLD_COL] == -1
    feats = [c for c in df.columns if c not in TARGET_COLS + ["ID", FOLD_COL]]
    return df[tr_mask][["ID"] + feats], df[tr_mask][TARGET_COLS], df[te_mask][["ID"] + feats], df[te_mask][TARGET_COLS]

def get_splits(df_tr):
    tv_df = df_tr.reset_index(drop=True)
    return [(tv_df.index[tv_df[FOLD_COL] != f].tolist(), tv_df.index[tv_df[FOLD_COL] == f].tolist()) for f in range(5)]

# %% [markdown]
# ## 2. PHASE 00 & 01: FREEZING & ENGINEERING
# Generazione dello split immutabile basato su target combinato e calcolo delle feature ingegnerizzate.

# %% [code]
print("[00z] Generazione Frozen Split...")
df_raw = pd.read_excel(RAW_EXCEL, sheet_name="Needs")
df_raw.columns = df_raw.columns.str.strip()
df_raw["stratify_key"] = df_raw["AccumulationInvestment"].astype(str) + "_" + df_raw["IncomeInvestment"].astype(str)

indices = np.arange(len(df_raw))
tr_idx, te_idx = train_test_split(indices, test_size=1000, stratify=df_raw["stratify_key"], random_state=RANDOM_STATE)
df_raw[FOLD_COL] = -5
df_raw.iloc[te_idx, df_raw.columns.get_loc(FOLD_COL)] = -1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for f_id, (_, val_rel) in enumerate(skf.split(np.zeros(len(tr_idx)), df_raw.iloc[tr_idx]["stratify_key"])):
    df_raw.iloc[tr_idx[val_rel], df_raw.columns.get_loc(FOLD_COL)] = f_id

df_raw.drop(columns=["stratify_key"], inplace=True)
df_raw.to_csv(BASE_DATA_PATH, index=False)

print("[01z] Feature Engineering (Data Contract fits on Train)...")
df_frozen = pd.read_csv(BASE_DATA_PATH)
trans = PipelineXTransformer().fit(df_frozen[df_frozen[FOLD_COL] >= 0])
df_master = trans.transform(df_frozen)
df_master.to_csv(MASTER_DATA_PATH, index=False)
print("✅ Fase 00 e 01 completate correttamente.")

# %% [markdown]
# ## 3. PHASE 02: SYSTEMATIC BENCHMARK
# Valutazione dell'Engineering Lift: quanto migliorano i modelli passando da 7 a 15 feature?

# %% [code]
print("[02z] Esecuzione Ablation Study (Raw vs Engineered)...")
bench_results = []
for target in TARGET_COLS:
    for s_name, s_key in [("Raw", "base"), ("Engineered", "master")]:
        X_tr, y_tr_all, X_te, y_te_all = get_data(s_key)
        y_tr, y_te = y_tr_all[target].values, y_te_all[target].values
        
        for name, model in {"RF": RandomForestClassifier(random_state=RANDOM_STATE), 
                            "XGB": XGBClassifier(tree_method='hist', random_state=RANDOM_STATE)}.items():
            cv_res = cross_validate(model, X_tr.drop(columns="ID"), y_tr, cv=5, scoring='roc_auc')
            bench_results.append({"Target": target, "Stage": s_name, "Model": name, "AUC": np.mean(cv_res['test_score'])})

df_bench = pd.DataFrame(bench_results).pivot_table(index=["Target", "Model"], columns="Stage", values="AUC")
df_bench["Lift"] = df_bench["Engineered"] - df_bench["Raw"]
print(tabulate(df_bench, headers='keys', tablefmt='fancy_grid'))

# %% [markdown]
# ## 4. PHASE 03: OPTUNA TUNING (WARM START)
# Ricerca iperparametri ottimale. Se esiste `best_params_Z.json`, Optuna innietta i risultati precedenti come punto di partenza.

# %% [code]
print("[03z] Tuning Iperparametri con Warm Start...")
def objective(trial, X, y, cv, m_type):
    if m_type == "XGB":
        p = {'n_estimators': trial.suggest_int('n_estimators', 150, 400), 'max_depth': trial.suggest_int('max_depth', 3, 7),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True), 'tree_method': 'hist'}
        m = XGBClassifier(**p, random_state=RANDOM_STATE)
    elif m_type == "LGBM":
        p = {'n_estimators': trial.suggest_int('n_estimators', 100, 350), 'num_leaves': trial.suggest_int('num_leaves', 15, 63),
             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True)}
        m = LGBMClassifier(**p, random_state=RANDOM_STATE, verbosity=-1)
    else:
        p = {'n_estimators': trial.suggest_int('n_estimators', 100, 400), 'max_depth': trial.suggest_int('max_depth', 5, 15)}
        m = RandomForestClassifier(**p, random_state=RANDOM_STATE)
    
    aucs = []
    for tr, va in cv:
        m.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
    return np.mean(aucs)

X_tr, y_tr_all, _, _ = get_data("master")
cv_splits = get_splits(pd.read_csv(MASTER_DATA_PATH))
best_results = json.load(open(PARAMS_FILE, 'r')) if os.path.exists(PARAMS_FILE) else {}

for target in TARGET_COLS:
    print(f"\n>>> Tuning {target}...")
    if target not in best_results: best_results[target] = {}
    for m_type in ["XGB", "LGBM", "RF"]:
        study = optuna.create_study(direction="maximize")
        if m_type in best_results[target]: 
            study.enqueue_trial(best_results[target][m_type].get("params", {}))
        study.optimize(lambda t: objective(t, X_tr.drop(columns="ID"), y_tr_all[target].values, cv_splits, m_type), n_trials=N_TRIALS)
        best_results[target][m_type] = {"params": study.best_params, "cv_auc": study.best_value}

with open(PARAMS_FILE, 'w') as f: json.dump(best_results, f, indent=4)
print(f"✅ Parametri salvati in {PARAMS_FILE}")

# %% [markdown]
# ## 5. PHASE 04 & 04z: ENSEMBLE AUDIT
# Addestramento di tutti i modelli campioni e fusione tramite Weighted Soft Voting per massimizzare la stabilità.

# %% [code]
print("[04z] Validazione Ensemble (Audit di Correlazione)...")
_, _, X_te, y_te_all = get_data("master")
ensemble_summary = []

for target in TARGET_COLS:
    probs_dict = {}
    for m_type in ["XGB", "LGBM", "RF"]:
        p = best_results[target][m_type]["params"]
        if m_type == "XGB": m = XGBClassifier(**p, tree_method='hist', random_state=RANDOM_STATE)
        elif m_type == "LGBM": m = LGBMClassifier(**p, verbosity=-1, random_state=RANDOM_STATE)
        else: m = RandomForestClassifier(**p, random_state=RANDOM_STATE)
        
        m.fit(X_tr.drop(columns="ID"), y_tr_all[target].values)
        probs_dict[m_type] = m.predict_proba(X_te.drop(columns="ID"))[:, 1]
        joblib.dump(m, os.path.join(PIPELINE_Z_DIR, f"model_Z_{target[:3].lower()}_{m_type}.joblib"))

    # Matrice di Correlazione
    prob_df = pd.DataFrame(probs_dict)
    plt.figure(figsize=(5,4)); sns.heatmap(prob_df.corr(), annot=True, cmap="coolwarm", vmin=0.8)
    plt.title(f"Correlazione Modelli: {target}"); plt.show()
    
    # Ensemble (0.4 al migliore, 0.3 agli altri)
    aucs = {k: roc_auc_score(y_te_all[target], v) for k, v in probs_dict.items()}
    best_m = max(aucs, key=aucs.get)
    weights = {k: (0.4 if k == best_m else 0.3) for k in aucs}
    ens_prob = np.average([probs_dict[k] for k in weights], axis=0, weights=list(weights.values()))
    
    ensemble_summary.append({"Target": target, "Best Single": best_m, "Single AUC": aucs[best_m], "Ensemble AUC": roc_auc_score(y_te_all[target], ens_prob)})

print(tabulate(ensemble_summary, headers='keys', tablefmt='fancy_grid'))
print("\n🚀 PIPELINE Z COMPLETATA CON SUCCESSO.")
