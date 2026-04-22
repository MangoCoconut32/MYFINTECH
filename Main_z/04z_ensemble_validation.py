"""
=============================================================================
04z_ensemble_validation.py — CORRELATION AUDIT & WEIGHTED ENSEMBLE
=============================================================================
PURPOSE:
    1. Check for redundancy: If models are too correlated (>0.95), ensembling
       is inefficient.
    2. Check for diversity: If models are diversely accurate (~0.85), 
       ensembling generates a "stability lift."
    3. Implement Weighted Soft Voting (0.4 Champion / 0.3 Others).
=============================================================================
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

# Double-Stage Pipeline Z Contract
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import utilsz

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
RANDOM_STATE = utilsz.RANDOM_STATE
TARGET_COLS  = utilsz.TARGET_COLS
OUT_DIR      = utilsz.PIPELINE_Z_DIR

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------------------------

def run_ensemble():
    print("=" * 80)
    print("04z — ENSEMBLE VALIDATION: CORRELATION & WEIGHTED VOTING")
    print("=" * 80)

    # 1. Load Blind Test Data
    X_te, y_te_all = utilsz.get_test_set(stage="master")
    f_cols = [c for c in X_te.columns if c != "ID"]
    X_te_df = X_te[f_cols]
    
    ensemble_results = []

    for target in TARGET_COLS:
        print(f"\n>>> [AUDIT] Target: {target}")
        y_te = y_te_all[target].values
        
        # 2. Extract Probabilities for fixed methods
        methods = ["XGB", "LGBM", "RF"]
        probs_dict = {}
        
        for m in methods:
            m_path = os.path.join(OUT_DIR, f"model_Z_{target.replace('Investment', '')}_{m}.joblib")
            if not os.path.exists(m_path):
                print(f"⚠️ Warning: Model {m} not found at {m_path}. Skipping.")
                continue
            
            model = joblib.load(m_path)
            # Ensure we pass the DataFrame if it's a GBDT
            probs_dict[m] = model.predict_proba(X_te_df)[:, 1]
        
        if len(probs_dict) < 2:
            print(f"❌ Error: Not enough models to create an ensemble for {target}.")
            continue
            
        prob_df = pd.DataFrame(probs_dict)
        
        # 3. Correlation Analysis
        corr_matrix = prob_df.corr()
        print(f"\n    [*] Correlation Matrix:")
        print(tabulate(corr_matrix, headers='keys', tablefmt='psql'))
        
        # 4. Visualization (Heatmap)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=0.8, vmax=1.0)
        plt.title(f"Model Probability Correlation - {target}", fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(OUT_DIR, f"corr_heatmap_{target}.png")
        plt.savefig(save_path)
        print(f"    ✅ Heatmap saved: {save_path}")
        plt.close()

        # 5. Determine Weights (Champion gets 0.4)
        individual_aucs = {m: roc_auc_score(y_te, p) for m, p in probs_dict.items()}
        sorted_methods = sorted(individual_aucs.items(), key=lambda x: x[1], reverse=True)
        
        weights = {}
        # User request: Champion 0.4, others 0.3/0.3
        weights[sorted_methods[0][0]] = 0.4
        weights[sorted_methods[1][0]] = 0.3
        weights[sorted_methods[2][0]] = 0.3
        
        print(f"\n    [*] Applying Weighted Soft Voting: {weights}")
        
        ens_prob = np.zeros_like(y_te, dtype=float)
        for m, w in weights.items():
            ens_prob += w * (probs_dict[m] / sum(weights.values())) # Normalized weights
            
        ens_auc = roc_auc_score(y_te, ens_prob)
        best_indiv_auc = sorted_methods[0][1]
        lift = ens_auc - best_indiv_auc
        
        print(f"    ⭐ Ensemble AUC: {ens_auc:.4f} (Lift over {sorted_methods[0][0]}: {lift:+.4f})")
        
        # Collect all rows for the final table
        for m in methods:
            ensemble_results.append({
                "Target": target.replace('Investment', ''),
                "Model": m,
                "Test AUC": round(individual_aucs[m], 4),
                "Type": "Individual"
            })
            
        ensemble_results.append({
            "Target": target.replace('Investment', ''),
            "Model": "ENSEMBLE",
            "Test AUC": round(ens_auc, 4),
            "Type": "Weighted Soft Voting"
        })

    # 6. Final Report
    print("\n" + "=" * 80)
    print("FINAL ENSEMBLE SUMMARY")
    print("=" * 80)
    print(tabulate(ensemble_results, headers='keys', tablefmt='fancy_grid', showindex=False))
    print("\n" + "=" * 80)
    print("✅ 04z COMPLETE: Ensemble audit finished.")
    print("=" * 80)

if __name__ == "__main__":
    run_ensemble()
