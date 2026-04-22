"""
=============================================================================
STEP 06 - SOTA MULTI-TASK LEARNING (MTL) NEURAL NETWORK
=============================================================================
PURPOSE:
    Break the ROC-AUC ceiling observed in Steps 03-05, particularly for the
    harder IncomeInvestment target (~0.760 with XGBoost).

    We build a Deep Learning model using the Keras Functional API that
    predicts BOTH targets simultaneously — this is called Multi-Task Learning.

WHY MULTI-TASK LEARNING (MTL)?
    Instead of training two separate models (one per target), we build a
    shared "trunk" network that learns a common feature representation,
    then branches into two independent output "heads".

    The key insight: by forcing the network to solve both prediction tasks
    at once, the shared trunk acts as a powerful regularizer. It cannot
    overfit to noise in one target because the same weights must also
    generalize for the other target.

ARCHITECTURE:
    Input -> [Dense -> BatchNorm -> GELU -> Dropout] × N layers (shared trunk)
          -> Head 1: Dense(1, sigmoid) -> AccumulationInvestment probability
          -> Head 2: Dense(1, sigmoid) -> IncomeInvestment probability

    The number of layers, units per layer, dropout rate, learning rate,
    and batch size are all searched by Optuna (30 trials).

WHY OPTUNA HERE TOO?
    Neural network architecture search (NAS) by hand is inefficient and
    biased by intuition. Optuna finds architecture + training hyperparameters
    jointly, which almost always beats manual search.

LOSS WEIGHTING:
    Income head gets a 1.5× loss weight vs 1.0× for Accumulation.
    This explicitly prioritizes improving the harder, lower-AUC target.

INPUTS:
    - Dataset2_Needs.xls  (Needs sheet, engineered features)
    - utils.py

OUTPUTS:
    - Output/06_neural_networks/06_neural_nets_results.csv          (precision, recall, F1, AUC per head)
    - Output/06_neural_networks/06_mtl_neural_net_weights.keras     (full model: architecture + weights)

    To reload the saved model:
        import tensorflow as tf
        model = tf.keras.models.load_model("Output/06_neural_networks/06_mtl_neural_net_weights.keras")
        preds = model.predict(X_new)
        prob_accumulation = preds[0].flatten()
        prob_income        = preds[1].flatten()
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import optuna
from tabulate import tabulate

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from utils import load_and_prepare_data

# Suppress Optuna per-trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.normpath(os.path.join(script_dir, "..", "Dataset2_Needs.xls"))

if not os.path.exists(FILE_PATH):
    print("Error: Could not find Dataset2_Needs.xls.")
    sys.exit(1)

print("=" * 100)
print("STEP 06: SOTA MULTI-TASK LEARNING (MTL) NEURAL NETWORK WITH OPTUNA")
print("=" * 100)

# ---------------------------------------------------------------------------
# 1. Data Loading Strategy
# ---------------------------------------------------------------------------
# We use utils.py to load Accumulation data (which handles the stratified
# split and scaling). Because X_train_df preserves the original DataFrame
# index, we can extract the matching Income labels from the raw Excel file
# using those same indices — this guarantees the two target vectors remain
# perfectly synchronized with the feature matrix.
X_train_df, X_test_df, y_train_acc, y_test_acc = load_and_prepare_data(
    FILE_PATH, 'AccumulationInvestment', use_engineered_features=True
)

# Read the raw file to extract the Income labels for the same row indices
needs_df = pd.read_excel(FILE_PATH, sheet_name='Needs')
y_train_inc = needs_df.loc[X_train_df.index, 'IncomeInvestment']
y_test_inc  = needs_df.loc[X_test_df.index,  'IncomeInvestment']

# Package targets as dictionaries — Keras multi-output models expect this format.
# The keys ('accumulo_out', 'income_out') must match the layer names defined below.
y_train_multi = {'accumulo_out': y_train_acc.values, 'income_out': y_train_inc.values}
y_test_multi  = {'accumulo_out': y_test_acc.values,  'income_out': y_test_inc.values}

# Create an internal validation split used exclusively by EarlyStopping.
# This split is carved from the training set and the test set remains untouched.
X_train_opt, X_val_opt, y_train_acc_opt, y_val_acc_opt, y_train_inc_opt, y_val_inc_opt = train_test_split(
    X_train_df, y_train_acc, y_train_inc, test_size=0.2, random_state=42
)
y_train_multi_opt = {'accumulo_out': y_train_acc_opt.values, 'income_out': y_train_inc_opt.values}
y_val_multi_opt   = {'accumulo_out': y_val_acc_opt.values,   'income_out': y_val_inc_opt.values}

# input_dim is fixed: it is the number of features after engineering
input_dim = X_train_df.shape[1]


# ---------------------------------------------------------------------------
# 2. Model Builder (Optuna Functional API)
# ---------------------------------------------------------------------------
def build_and_compile_model(trial):
    """
    Builds and compiles the MTL network using hyperparameters proposed by Optuna.

    The architecture has two parts:
      TRUNK: Shared layers that learn features useful for both targets.
             Optuna controls the number of layers (1-3) and size of each.
      HEADS: Two separate output neurons, one per target.
             They share the trunk's learned representation but have independent weights.

    BatchNormalization: normalizes layer outputs to speed up training and
                        reduce sensitivity to weight initialization.
    GELU activation:    smoother than ReLU; performs well on structured tabular data.
    Dropout:            randomly zeros out neurons during training to prevent co-adaptation
                        and force the network to learn redundant representations.
    AdamW optimizer:    Adam with decoupled weight decay — better regularization than Adam.
    """
    inputs = Input(shape=(input_dim,))
    x = inputs

    # --- TRUNK: Shared Representation ---
    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        units        = trial.suggest_int(f'n_units_l{i}', 32, 128)
        dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)

        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        x = Dropout(dropout_rate)(x)

    # --- HEADS: Task-Specific Output Branches ---
    # Sigmoid activation maps the output to [0, 1] — the predicted probability
    out_acc = Dense(1, activation='sigmoid', name='accumulo_out')(x)
    out_inc = Dense(1, activation='sigmoid', name='income_out')(x)

    model = Model(inputs=inputs, outputs=[out_acc, out_inc])

    # --- Optimizer ---
    lr        = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer = AdamW(learning_rate=lr)

    # --- Loss Weights ---
    # Income gets 1.5× weight because it had lower AUC in Steps 03-05.
    # This biases gradient updates to improve the harder target more aggressively.
    model.compile(
        optimizer=optimizer,
        loss={'accumulo_out': 'binary_crossentropy', 'income_out': 'binary_crossentropy'},
        loss_weights={'accumulo_out': 1.0, 'income_out': 1.5}
    )
    return model


def objective(trial):
    """
    Optuna objective: builds a trial model, trains it, and returns the validation loss.
    Lower loss = better. Optuna will minimize this value across N trials.

    clear_session() frees GPU/CPU memory between trials to prevent memory leaks
    when many Keras models are built in sequence.
    """
    tf.keras.backend.clear_session()

    model      = build_and_compile_model(trial)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    callbacks = [
        # Stop training if validation loss doesn't improve for 15 epochs.
        # restore_best_weights: roll back to the epoch with the lowest val_loss.
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        # If val_loss plateaus for 5 epochs, halve the learning rate.
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
    ]

    history = model.fit(
        X_train_opt, y_train_multi_opt,
        validation_data=(X_val_opt, y_val_multi_opt),
        epochs=100,       # maximum epochs (EarlyStopping will cut this short if needed)
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0         # silent training during search
    )

    # Return the best (lowest) validation loss achieved during training.
    # This is a weighted sum of both heads' losses, adjusted by loss_weights.
    return min(history.history['val_loss'])


# ---------------------------------------------------------------------------
# 3. Optuna Search
# ---------------------------------------------------------------------------
print("\n[1] Commencing Bayesian Architecture Search on CPU (30 Trials)...")
study = optuna.create_study(direction="minimize")  # minimize val_loss
study.optimize(objective, n_trials=30, show_progress_bar=False)

print("\nBest Hyperparameters Discovered:", study.best_params)

# ---------------------------------------------------------------------------
# 4. Final Training with Best Configuration
# ---------------------------------------------------------------------------
# Re-build and re-train using the winning parameters from the search.
# clear_session() ensures no trial-specific state carries over.
print("\n[2] Training Golden SOTA Configuration with Best Parameters...")
tf.keras.backend.clear_session()
best_model      = build_and_compile_model(study.best_trial)
final_batch_size = study.best_trial.params['batch_size']

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

best_model.fit(
    X_train_opt, y_train_multi_opt,
    validation_data=(X_val_opt, y_val_multi_opt),
    epochs=100,
    batch_size=final_batch_size,
    callbacks=callbacks,
    verbose=0
)

# ---------------------------------------------------------------------------
# 5. Model Persistence
# ---------------------------------------------------------------------------
# The .keras format saves the complete model: architecture, weights,
# optimizer state. It can be fully restored with tf.keras.models.load_model().
print("\n[3] Saving SOTA Neural Network Weights...")
model_save_path = os.path.normpath(os.path.join(script_dir, "..", "Output", "06_neural_networks", "06_mtl_neural_net_weights.keras"))
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
best_model.save(model_save_path)
print(f"Model saved to: {model_save_path}")

# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
print("\n[4] Generating Formal Predictions on Hold-Out Test Set...")
preds = best_model.predict(X_test_df, verbose=0)
# preds[0] = accumulation head output, preds[1] = income head output
# flatten() converts shape (N,1) to (N,) for scikit-learn compatibility
pred_acc_prob = preds[0].flatten()
pred_inc_prob = preds[1].flatten()

# Convert probabilities to binary class labels using 0.5 threshold
pred_acc_class = (pred_acc_prob > 0.5).astype(int)
pred_inc_class = (pred_inc_prob > 0.5).astype(int)

all_nn_results = []

# Evaluate each head independently against its true labels
for target_col, prob, cls_pred, true_vals in [
    ("AccumulationInvestment", pred_acc_prob, pred_acc_class, y_test_acc),
    ("IncomeInvestment",       pred_inc_prob, pred_inc_class, y_test_inc)
]:
    prec = precision_score(true_vals, cls_pred, zero_division=0)
    rec  = recall_score(true_vals, cls_pred, zero_division=0)
    f1   = f1_score(true_vals, cls_pred, zero_division=0)
    # ROC-AUC uses raw probabilities (not class labels) — it measures ranking quality
    roc  = roc_auc_score(true_vals, prob)

    all_nn_results.append({
        "Algorithm":    "SOTA MTL Neural Network",
        "Target":       target_col,
        "Test Precision": f"{prec:.3f}",
        "Test Recall":    f"{rec:.3f}",
        "Test F1":        f"{f1:.3f}",
        "Test ROC-AUC":   f"{roc:.3f}"
    })

# ---------------------------------------------------------------------------
# Output: Console + CSV
# ---------------------------------------------------------------------------
df_nn = pd.DataFrame(all_nn_results)
output_dir = os.path.normpath(os.path.join(script_dir, "..", "Output", "06_neural_networks"))
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "06_neural_nets_results.csv")
df_nn.to_csv(csv_path, index=False)

print("\n" + "=" * 120)
print("STEP 06: NEURAL NETWORK MASTER TABLE")
print("=" * 120)
print(tabulate(df_nn, headers='keys', tablefmt='grid', showindex=False))
print(f"\nPM Report saved to: {csv_path}")
