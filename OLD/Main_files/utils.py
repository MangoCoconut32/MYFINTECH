import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

def load_and_prepare_data(filepath, target_col, use_engineered_features=True):
    """
    Loads data from Excel, performs optional feature engineering, normalizes features, 
    and splits into train and test sets to avoid data leakage.
    
    Parameters:
    -----------
    filepath : str
        Relative or absolute path to the Dataset2_Needs.xls file.
    target_col : str
        The name of the target variable to predict (e.g., 'AccumulationInvestment').
    use_engineered_features : bool
        If True, applies Dev 1's custom feature transformations. If False, skips them.
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : pandas.DataFrames and pandas.Series
        The fully processed and scaled datasets ready for machine learning models.
    """
    # Load dataset
    needs_df = pd.read_excel(filepath, sheet_name='Needs')
    
    # Strip whitespace from column names to prevent KeyError bugs later
    needs_df.columns = needs_df.columns.str.strip()
    
    # Drop the ID column as it contains no predictive value and can confuse models
    if 'ID' in needs_df.columns:
        needs_df = needs_df.drop('ID', axis=1)
        
    # Split features (X) and target (y). We drop both potential targets from X 
    # to prevent one target from leaking information into the other model.
    X = needs_df.drop(columns=['IncomeInvestment', 'AccumulationInvestment'])
    y = needs_df[target_col]
    
    # -------------------------------------------------------------
    # Feature Engineering Experiments (from Phase 1 / Dev 1)
    # -------------------------------------------------------------
    if use_engineered_features:
        # Log transformations: Financial variables usually follow a long-tailed 
        # (Pareto-like) distribution. Applying a log transform helps normalize 
        # the distribution and reduces the outsized impact of ultra-wealthy outliers.
        X['Wealth_log'] = np.log1p(X['Wealth'])
        X['Income_log'] = np.log1p(X['Income'])
        
        # Per-member metrics: Absolute wealth means less than usable wealth per person.
        X['Wealth_per_person'] = X['Wealth'] / X['FamilyMembers']
        X['Income_per_person'] = X['Income'] / X['FamilyMembers']
        
        # Income to Wealth ratio: A proxy for a client's life cycle. High ratio usually 
        # implies younger accumulation-phase clients, low ratio implies older income-phase.
        X['Inc_to_Wealth_ratio'] = X['Income'].div(X['Wealth'].replace(0, np.nan))
        X['Inc_to_Wealth_ratio'] = X['Inc_to_Wealth_ratio'].fillna(X['Income'].max())
        
        # Age brackets: Binning ages helps capture non-linear relationships.
        X['Age_bracket'] = pd.cut(X['Age'], bins=[17, 35, 55, 100], labels=['Young', 'Mid', 'Senior'])
        
        # One-hot encode the Age Bracket because most Scikit-Learn models 
        # require strictly numerical inputs.
        X = pd.get_dummies(X, columns=['Age_bracket'], drop_first=False, dtype=int)

    
    # -------------------------------------------------------------
    # Train / Test Split
    # -------------------------------------------------------------
    # We execute the split BEFORE scaling. This ensures that the Test set remains 
    # truly unseen data, completely isolating our evaluation metrics.
    # stratify=y guarantees that the ratio of True/False targets remains exactly the same.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Make explicit deep copies to satisfy Pandas and avoid SettingWithCopyWarning
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # -------------------------------------------------------------
    # Normalization (Fit ONLY on Train!)
    # -------------------------------------------------------------
    # MinMaxScaler ensures all numeric inputs live between 0 and 1, vastly 
    # improving convergence speed for gradient descent and distance calculations (KNN).
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    
    # Important: .fit_transform on Training set, but ONLY .transform on Test set!
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    return X_train, X_test, y_train, y_test


def get_all_engineered_features(filepath):
    """
    Loads the full Needs sheet, applies the same feature engineering pipeline
    as load_and_prepare_data (with use_engineered_features=True), and returns
    the COMPLETE feature matrix alongside both target columns.

    Unlike load_and_prepare_data, this function does NOT split into train/test
    and does NOT apply MinMaxScaling. It is intended for INFERENCE on all clients
    (e.g., in the recommender system), where we need a propensity score for every
    row and evaluation metrics are not required.

    XGBoost is scale-invariant (tree splits are rank-based), so scaling is not
    needed for correct predictions — only the feature names and engineering steps
    must exactly match what was used during training.

    Parameters:
    -----------
    filepath : str
        Absolute or relative path to Dataset2_Needs.xls.

    Returns:
    --------
    X : pandas.DataFrame
        All 15 engineered features for every client.
    y_acc : pandas.Series
        Ground-truth AccumulationInvestment labels (used for coverage diagnostics).
    y_inc : pandas.Series
        Ground-truth IncomeInvestment labels.
    needs_df_full : pandas.DataFrame
        The full Needs sheet (with ID, targets, and raw columns) for downstream
        joining of predicted scores back to client metadata.
    """
    # Load and strip column names (the Excel file has a trailing space on 'Income ')
    needs_df = pd.read_excel(filepath, sheet_name='Needs')
    needs_df.columns = needs_df.columns.str.strip()

    # Preserve the full dataframe for downstream use before dropping anything
    needs_df_full = needs_df.copy()

    # Isolate features — drop ID and both targets to prevent leakage
    X = needs_df.drop(columns=['ID', 'IncomeInvestment', 'AccumulationInvestment'])

    # Extract targets separately for diagnostic use
    y_acc = needs_df['AccumulationInvestment']
    y_inc = needs_df['IncomeInvestment']

    # -----------------------------------------------------------------------
    # Apply identical feature engineering as load_and_prepare_data
    # Any change to the engineering block there MUST be mirrored here.
    # -----------------------------------------------------------------------
    X['Wealth_log'] = np.log1p(X['Wealth'])
    X['Income_log'] = np.log1p(X['Income'])
    X['Wealth_per_person'] = X['Wealth'] / X['FamilyMembers']
    X['Income_per_person'] = X['Income'] / X['FamilyMembers']
    X['Inc_to_Wealth_ratio'] = X['Income'].div(X['Wealth'].replace(0, np.nan))
    X['Inc_to_Wealth_ratio'] = X['Inc_to_Wealth_ratio'].fillna(X['Income'].max())
    X['Age_bracket'] = pd.cut(X['Age'], bins=[17, 35, 55, 100], labels=['Young', 'Mid', 'Senior'])
    X = pd.get_dummies(X, columns=['Age_bracket'], drop_first=False, dtype=int)

    return X, y_acc, y_inc, needs_df_full


def evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds=5):
    """
    Evaluates a given machine learning model structure. 
    If cv_folds > 1, it performs Stratified k-fold cross-validation on the training set.
    
    Returns:
    --------
    results : dict
        A dictionary containing the CV metrics, the Final Test metrics, and the model itself.
    """
    results = {
        'cv_metrics': None,
        'test_metrics': {},
        'model': model
    }
    
    # Part 1: Cross Validation
    # We skip this if cv_folds is 0 or 1, which is useful when training massive Neural Networks.
    if cv_folds > 1:
        # We use StratifiedKFold rather than raw KFold to ensure target classes are evenly distributed.
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            # Isolate the training and validation slices for this fold
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            # Train the model on the remaining 4/5ths of data
            model.fit(X_train_fold, y_train_fold)
            
            # Predict the 1/5th validation slice
            y_val_pred = model.predict(X_val_fold)
            
            # ROC-AUC requires probabilities rather than absolute 0/1 predictions
            y_val_prob = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, 'predict_proba') else y_val_pred
            
            # Append metrics to calculate Mean and Standard Deviation later
            cv_metrics['precision'].append(precision_score(y_val_fold, y_val_pred, zero_division=0))
            cv_metrics['recall'].append(recall_score(y_val_fold, y_val_pred, zero_division=0))
            cv_metrics['f1'].append(f1_score(y_val_fold, y_val_pred, zero_division=0))
            cv_metrics['roc_auc'].append(roc_auc_score(y_val_fold, y_val_prob))
            
        # Aggregate the 5 metrics into mean and std
        results['cv_metrics'] = {
            metric: {'mean': np.mean(scores), 'std': np.std(scores)} 
            for metric, scores in cv_metrics.items()
        }
        
    # Part 2: Final Fit & Test Set Evaluation
    # Since CV is just for scoring, we retrain the model exactly once on the ENTIRE training set.
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_test_pred
    
    # Calculate raw scores against the isolated 20% test slice
    results['test_metrics'] = {
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_test_prob)
    }
    
    return results


def display_results(results_dict, model_name):
    """
    Takes the output of evaluate_model() and displays it automatically formatted.
    """
    # Format CV if it was executed
    if results_dict['cv_metrics'] is not None:
        cv_data = {
            'Metric': ['Precision', 'Recall', 'F1', 'ROC-AUC'],
            'CV Mean': [
                results_dict['cv_metrics']['precision']['mean'],
                results_dict['cv_metrics']['recall']['mean'],
                results_dict['cv_metrics']['f1']['mean'],
                results_dict['cv_metrics']['roc_auc']['mean']
            ],
            'CV Std': [
                results_dict['cv_metrics']['precision']['std'],
                results_dict['cv_metrics']['recall']['std'],
                results_dict['cv_metrics']['f1']['std'],
                results_dict['cv_metrics']['roc_auc']['std']
            ],
            'Test Set': [
                results_dict['test_metrics']['precision'],
                results_dict['test_metrics']['recall'],
                results_dict['test_metrics']['f1'],
                results_dict['test_metrics']['roc_auc']
            ]
        }
    else:
        # Fallback format for single-shot evaluations (like Deep NNs)
        cv_data = {
            'Metric': ['Precision', 'Recall', 'F1', 'ROC-AUC'],
            'Test Set': [
                results_dict['test_metrics']['precision'],
                results_dict['test_metrics']['recall'],
                results_dict['test_metrics']['f1'],
                results_dict['test_metrics']['roc_auc']
            ]
        }
    
    # Display table safely using Pandas and Tabulate
    df = pd.DataFrame(cv_data).round(3)
    
    print(f"\n{model_name}")
    print("=" * 60)
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
