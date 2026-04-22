"""Explainable Boosting Machine - InterpretML's glassbox model.

EBMs are GA2Ms: a learned shape function f_j(x_j) per feature plus a few
pairwise interactions f_jk(x_j, x_k). The whole model is interpretable
by construction, so we don't need SHAP to explain it.

Compared against the Optuna XGB ceiling, then the global dashboard is
exported to an HTML file.
"""
import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_and_prepare_data

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show as _interpret_show                                     
from interpret.provider import InlineProvider
import interpret
interpret.set_visualize_provider(InlineProvider())                       

_script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(_script_dir, "..", "Dataset2_Needs.xls"))
MODEL_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "04_optuna"))
OUT_DIR = os.path.normpath(os.path.join(_script_dir, "..", "Output", "05_ensembles"))
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
for target in ["AccumulationInvestment", "IncomeInvestment"]:
    print(f"\n--- TARGET: {target} ---")
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        FILE_PATH, target, use_engineered_features=True
    )


    print("  Training Explainable Boosting Machine...")
    ebm = ExplainableBoostingClassifier(
        interactions=10, learning_rate=0.02, max_bins=256,
        random_state=42, n_jobs=-1,
    )
    ebm.fit(X_train, y_train)

    p_ebm = ebm.predict_proba(X_test)[:, 1]
    pred_ebm = (p_ebm >= 0.5).astype(int)
    auc_ebm = roc_auc_score(y_test, p_ebm)


    with open(f"{MODEL_DIR}/04_optuna_{target}_xgb.pkl", "rb") as f:
        xgb = pickle.load(f)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, p_xgb)

    print(f"  EBM AUC = {auc_ebm:.4f}   |   XGB-Optuna AUC = {auc_xgb:.4f}   |   Δ = {auc_ebm - auc_xgb:+.4f}")
    rows.append({
        "Target": target,
        "Model": "EBM (glassbox)",
        "Test_ROC_AUC": round(auc_ebm, 4),
        "Test_Precision": round(precision_score(y_test, pred_ebm), 4),
        "Test_Recall": round(recall_score(y_test, pred_ebm), 4),
        "Test_F1": round(f1_score(y_test, pred_ebm), 4),
        "XGB_Optuna_AUC_reference": round(auc_xgb, 4),
        "Delta_vs_XGB": round(auc_ebm - auc_xgb, 4),
    })


    print("  Exporting global explanation dashboard...")
    global_exp = ebm.explain_global(name=f"EBM Global - {target}")
    html_path = os.path.join(OUT_DIR, f"05b_ebm_global_{target}.html")
    try:
        viz = global_exp.visualize()
        if hasattr(viz, "to_html"):
            with open(html_path, "w") as fh:
                fh.write(viz.to_html(full_html=True, include_plotlyjs="cdn"))
        else:
            # InterpretML a volte restituisce un oggetto Dash, questa è una fallback sicura
            import json
            with open(html_path, "w") as fh:
                fh.write("<html><body><pre>")
                fh.write(json.dumps(global_exp.data(), indent=2, default=str))
                fh.write("</pre></body></html>")
    except Exception as e:
        print(f"    HTML export failed ({e}); skipping but model is saved.")
    print(f"    Saved Dashboard: {html_path}")

    # ==========================================
    # FIX: SALVATAGGIO FISICO DEL MODELLO
    # ==========================================
    model_path = os.path.join(OUT_DIR, f"05b_ebm_{target}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(ebm, f)
    print(f"    Saved Model Weights: {model_path}")

# Fuori dal ciclo for, salva il CSV...
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "05b_ebm_results.csv"), index=False)
print(f"\nSaved Metrics: {OUT_DIR}/05b_ebm_results.csv")
print(df.to_string(index=False))
