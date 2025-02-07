#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter

def preprocess_data(df, imputer=None, fit_imputer=False):
    """Handle missing values and outliers"""
    df = df.copy()
    
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    
    # Handle missing values and zeros
    df['MonthlyIncome'] = df['MonthlyIncome'].replace(0, np.nan)
    df['NumberOfDependents'] = df['NumberOfDependents'].replace(0, np.nan)
    
    # Handle unrealistic values in past due columns
    past_due_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse'
    ]
    
    for col in past_due_cols:
        df[col] = np.where(df[col] > 20, np.nan, df[col])
    
    # Impute missing values
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        
    if fit_imputer:
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    else:
        df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
    
    return df_imputed, imputer

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    metrics = {}
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred)
    metrics['f1'] = f1_score(y_test, y_pred)
    metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['confusion_matrix'] = {
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp
    }
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'feature_importance': get_feature_importance(model, X_test.columns) 
        if hasattr(model, 'feature_importances_') else None
    }

def get_feature_importance(model, feature_names):
    """Get feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    return None

def train_logistic_regression(X_train, y_train):
    """Logistic Regression with handling for class imbalance"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            random_state=42,
            solver='lbfgs'
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_random_forest(X_train, y_train):
    """Random Forest with balanced class weights"""
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """XGBoost with handling for class imbalance"""
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def prepare_submission(model, test_df, imputer, feature_columns, output_file):
    """Generate submission file in required format"""
    # Preprocess test data
    X_test, _ = preprocess_data(test_df, imputer=imputer, fit_imputer=False)
    
    # Ensure correct feature order
    X_test = X_test[feature_columns]
    
    # Predict probabilities
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'Id': test_df['Unnamed: 0'],
        'Probability': probabilities
    })
    
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

if __name__ == "__main__":
    # ... [previous code] ...

    evaluation_results = {}
    
    for name, trainer in models.items():
        print(f"\nTraining {name}...")
        model = trainer(X_res, y_res)
        evaluation = evaluate_model(model, X_val, y_val, name)
        evaluation_results[name] = {
            'model': model,
            'metrics': evaluation['metrics'],
            'model_name': name
        }
        print(f"{name} Validation ROC-AUC: {evaluation['metrics']['roc_auc']:.4f}")
    
    # Select best model based on ROC-AUC
    best_model_name = max(evaluation_results, 
                         key=lambda x: evaluation_results[x]['metrics']['roc_auc'])
    best_model_info = evaluation_results[best_model_name]
    best_model = best_model_info['model']
    
    print(f"\nBest model: {best_model_info['model_name']}")
    
    # Prepare final submission
    feature_columns = X_train_clean.columns.tolist()
    prepare_submission(
        best_model,
        test_df,
        imputer,
        feature_columns,
        'final_submission.csv'
    )