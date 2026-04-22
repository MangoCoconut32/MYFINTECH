"""
=============================================================================
STEP 08 - KNOWLEDGE-BASED RECOMMENDER SYSTEM
=============================================================================
PURPOSE:
    This is the final operational layer of the pipeline. It loads the
    Optuna-tuned XGBoost models from Step 04 and uses them to generate
    propensity scores that drive concrete, regulation-compliant product
    recommendations for every client.

HOW IT FITS IN THE PIPELINE:
    Steps 02-06 answered: "Does this client need Accumulation or Income?"
    Step 08 answers:      "Which specific product should we offer them, and why?"

THE PREDICTION STRATEGY:
    We load the Optuna-tuned XGBoost models serialized by Step 04 and use
    them to generate propensity scores for ALL clients. This ensures the
    recommender uses the exact same models that were validated and audited
    in Steps 04-07, not a separately trained approximation.

THE FOUR RECOMMENDATION RULES:

  "strict":    Regulatory-compliant. Only recommends products whose risk level
               does not exceed the client's MIFID risk propensity score.
               Among valid options, maximizes the product risk (best return potential).
               May return None if no compliant product exists.

  "closest":   Business-maximizing. Ignores the hard risk boundary and simply
               finds the product whose risk is numerically closest to the client's
               propensity. Guarantees 100% coverage but may violate strict compliance.

  "top3":      Like "closest" but returns the 3 nearest products, enabling
               a human advisor to make the final selection from a shortlist.

  "age_gated": Regulatory override for elderly clients. If a client is Income-seeking
               AND over 65 years old, their effective risk ceiling is capped at 0.4
               (a conservative maximum), then "strict" logic is applied.
               This simulates prudential suitability rules for vulnerable clients.

COVERAGE METRICS:
    Coverage = % of clients who received at least one recommendation.
    "strict" will have lower coverage (some clients have no compliant product).
    "closest" will always be 100%.
    This trade-off is the core business decision executives must make.

INPUTS:
    - Dataset2_Needs.xls  (Needs sheet for client data, Products sheet for catalogue)
    - Output/04_optuna/04_optuna_AccumulationInvestment_xgb.pkl   (best XGB from Step 04)
    - Output/04_optuna/04_optuna_IncomeInvestment_xgb.pkl         (best XGB from Step 04)

OUTPUTS:
    - Output/08_recommender/08_recommender_coverage.csv      (coverage % per rule — for PM)
    - Output/08_recommender/08_client_recommendations.csv    (full per-client recommendation table)
=============================================================================
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from tabulate import tabulate

# utils.py lives in the same directory as this script.
# Inserting the script directory ensures it can be found regardless of
# which working directory the caller uses to invoke this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_all_engineered_features

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH  = os.path.normpath(os.path.join(script_dir, "..", "Dataset2_Needs.xls"))

if not os.path.exists(FILE_PATH):
    print("Error: Could not find Dataset2_Needs.xls.")
    sys.exit(1)

print("=" * 100)
print("STEP 08: KNOWLEDGE-BASED RECOMMENDER SYSTEM")
print("=" * 100)

# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------
print("\n[1] Loading client and product data...")

# Load the product catalogue separately (it has no features to engineer)
products_df = pd.read_excel(FILE_PATH, sheet_name='Products')

# get_all_engineered_features() applies the same engineering as load_and_prepare_data:
#   - strips the Excel trailing space on 'Income ' -> 'Income'
#   - log transforms, per-member ratios, Inc_to_Wealth_ratio, age bracket dummies
# It returns (X, y_acc, y_inc, needs_df_full) where needs_df_full retains the
# original columns (ID, targets, raw fields) for later client metadata stitching.
# XGBoost is scale-invariant (tree splits are threshold-based), so the MinMaxScaler
# from training is NOT required for inference — only feature names must match.
X, y_acc, y_inc, needs_df = get_all_engineered_features(FILE_PATH)
print(f" -> Engineered feature matrix: {X.shape[0]} clients \u00d7 {X.shape[1]} features")

# ---------------------------------------------------------------------------
# 2. Model Loading from Step 04 (Bayesian Optuna)
# ---------------------------------------------------------------------------
# Load the Optuna-tuned XGBoost models serialized by Step 04.
# Using the fitted models from Step 04 ensures the propensity scores reflect
# the exact same model that was optimized, explained (Step 07), and validated.
# The models were trained on the stratified train split — here we call predict_proba
# on the FULL dataset to generate an operational score for every client.
optuna_dir = os.path.normpath(os.path.join(script_dir, "..", "Output", "04_optuna"))

for pkl_name in [
    "04_optuna_AccumulationInvestment_xgb.pkl",
    "04_optuna_IncomeInvestment_xgb.pkl"
]:
    if not os.path.exists(os.path.join(optuna_dir, pkl_name)):
        print(f"ERROR: Required model not found: {os.path.join(optuna_dir, pkl_name)}")
        print("Please run 04_bayesian_optuna.py before running this script.")
        sys.exit(1)

print(" -> Loading XGBoost models from Step 04 output...")
xgb_acc = joblib.load(os.path.join(optuna_dir, "04_optuna_AccumulationInvestment_xgb.pkl"))
xgb_inc = joblib.load(os.path.join(optuna_dir, "04_optuna_IncomeInvestment_xgb.pkl"))
print(" -> Models loaded successfully.")

# ---------------------------------------------------------------------------
# 3. Need Assignment
# ---------------------------------------------------------------------------
print("\n[2] Generating propensity scores and assigning predicted needs...")

# predict_proba returns [[P(class=0), P(class=1)]] — we take column 1 (probability of need=1)
prob_acc = xgb_acc.predict_proba(X)[:, 1]
prob_inc = xgb_inc.predict_proba(X)[:, 1]

# Attach propensity scores back to needs_df (which includes ID, Age, RiskPropensity
# and all original columns — useful for the downstream per-client export at section 6).
needs_df['Prob_Accumulation'] = prob_acc
needs_df['Prob_Income']       = prob_inc

# Assign the dominant need: whichever target has the higher predicted probability.
# This is a winner-takes-all assignment — a client gets ONE primary need.
needs_df['Predicted_Need'] = np.where(prob_acc > prob_inc, "Accumulation", "Income")

print(f" -> Predicted needs assigned to {len(needs_df)} clients.")


# ---------------------------------------------------------------------------
# 4. Recommender Engine Core Function
# ---------------------------------------------------------------------------

def match_product(client_row, products_df, rule="strict"):
    """
    Given a single client row and the product catalogue, return the best product.

    Parameters:
        client_row  : pandas Series — one row from needs_df (includes Predicted_Need,
                      RiskPropensity, Age, etc.)
        products_df : pandas DataFrame — the Products sheet (IDProduct, Type, Risk)
        rule        : str — one of 'strict', 'closest', 'top3', 'age_gated'

    Returns:
        int   — single ProductID (for 'strict', 'closest', 'age_gated')
        list  — list of up to 3 ProductIDs (for 'top3')
        None  — if no valid product exists (only possible with 'strict' rule)
    """
    # Map textual need to the Products sheet binary convention:
    #   1 = Accumulation product, 0 = Income product  (from Metadata sheet)
    need_type   = 1 if client_row['Predicted_Need'] == "Accumulation" else 0
    client_risk = client_row['RiskPropensity']
    client_age  = client_row['Age']

    # --- Age-Gating Override ---
    # For elderly Income-seeking clients (age > 65), cap their effective risk
    # tolerance at 0.4 to enforce prudential suitability rules.
    # After capping, we fall through to the standard 'strict' logic.
    if rule == "age_gated":
        if client_row['Predicted_Need'] == "Income" and client_age > 65:
            client_risk = min(client_risk, 0.4)  # conservative hard cap
        rule = "strict"  # process the rest as strict

    # Filter the catalogue to only products matching the client's predicted need type
    filtered_products = products_df[products_df['Type'] == need_type].copy()

    # Edge case: if the catalogue has no products of this type at all
    if filtered_products.empty:
        return [] if rule == "top3" else None

    # --- Strict Rule ---
    if rule == "strict":
        # Keep only products whose risk is AT OR BELOW the client's propensity.
        # This is the MIFID suitability constraint: never offer a riskier product
        # than the client's documented risk tolerance.
        valid = filtered_products[filtered_products['Risk'] <= client_risk]
        if valid.empty:
            return None  # no compliant product exists — this client is uncovered
        # Among compliant products, select the one with the highest risk.
        # Higher risk = higher expected return = best outcome for a willing client.
        best_product = valid.sort_values(by='Risk', ascending=False).iloc[0]['IDProduct']
        return int(best_product)

    # --- Closest Rule ---
    elif rule == "closest":
        # Ignore the hard risk boundary. Find the product whose risk is
        # numerically closest to the client's propensity (minimizes |delta|).
        # This maximizes coverage at the cost of possible regulatory non-compliance.
        filtered_products['Risk_Penalty'] = (filtered_products['Risk'] - client_risk).abs()
        best_product = filtered_products.sort_values(by='Risk_Penalty').iloc[0]['IDProduct']
        return int(best_product)

    # --- Top 3 Rule ---
    elif rule == "top3":
        # Same as 'closest' but return the 3 best products as a shortlist.
        # Useful for human advisors who want options rather than a single recommendation.
        filtered_products['Risk_Penalty'] = (filtered_products['Risk'] - client_risk).abs()
        top_products = (
            filtered_products.sort_values(by='Risk_Penalty')
            .head(3)['IDProduct']
            .tolist()
        )
        return [int(x) for x in top_products]

    return None  # fallthrough safety — should never be reached


# ---------------------------------------------------------------------------
# 5. Coverage Evaluation
# ---------------------------------------------------------------------------
print("\n[3] Computing coverage statistics for all four recommendation rules...")

rule_sets      = ["strict", "closest", "top3", "age_gated"]
coverage_stats = []

for active_rule in rule_sets:
    valid_matches = 0

    for idx, row in needs_df.iterrows():
        recommendation = match_product(row, products_df, rule=active_rule)

        # Count a match as valid if:
        #   - A single integer ProductID was returned  (strict / closest / age_gated)
        #   - A non-empty list was returned            (top3)
        if recommendation is not None:
            if isinstance(recommendation, list) and len(recommendation) > 0:
                valid_matches += 1
            elif isinstance(recommendation, int):
                valid_matches += 1

    coverage_pct = (valid_matches / len(needs_df)) * 100
    coverage_stats.append({
        "Engine Logic Rule":    active_rule,
        "Total Clients Matched": valid_matches,
        "Coverage Portfolio %":  f"{coverage_pct:.2f}%"
    })

df_coverage = pd.DataFrame(coverage_stats)

print("\n" + "=" * 80)
print("RECOMMENDER ENGINE: COVERAGE & COMPLIANCE EVALUATION")
print("=" * 80)
print(tabulate(df_coverage, headers='keys', tablefmt='grid', showindex=False))
print("\nINTERPRETATION:")
print("  'strict'    -> % of clients with a fully MIFID-compliant product available")
print("  'closest'   -> always 100%; maximizes sales but may exceed risk boundary")
print("  'top3'      -> always 100%; provides shortlist for human advisor review")
print("  'age_gated' -> strict coverage after applying extra protection for age>65")

# ---------------------------------------------------------------------------
# 6. Export
# ---------------------------------------------------------------------------
output_dir = os.path.normpath(os.path.join(script_dir, "..", "Output", "08_recommender"))
os.makedirs(output_dir, exist_ok=True)

# Save the coverage comparison table for PM reporting
coverage_path = os.path.join(output_dir, "08_recommender_coverage.csv")
df_coverage.to_csv(coverage_path, index=False)
print(f"\nCoverage Report saved to: {coverage_path}")

# Generate the full per-client recommendation table using 'closest' (100% coverage).
# This is the operational output — one row per client with their recommended product.
print("\n[4] Generating full per-client recommendation table (rule='closest')...")
rec_rows = []
for idx, row in needs_df.iterrows():
    rec = match_product(row, products_df, rule="closest")
    rec_rows.append({
        "ClientID":              int(row['ID']),
        "Predicted_Need":        row['Predicted_Need'],
        "Prob_Accumulation":     f"{row['Prob_Accumulation']:.3f}",
        "Prob_Income":           f"{row['Prob_Income']:.3f}",
        "RiskPropensity":        row['RiskPropensity'],
        "Recommended_ProductID": rec
    })

df_recommendations = pd.DataFrame(rec_rows)
rec_path = os.path.join(output_dir, "08_client_recommendations.csv")
df_recommendations.to_csv(rec_path, index=False)
print(f"Per-Client Recommendations saved to: {rec_path}")

print("\n[Pipeline Complete] Step 08 finished. Ready for production integration.")
