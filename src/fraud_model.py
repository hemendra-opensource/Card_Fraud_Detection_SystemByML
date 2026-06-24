import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from src.utils import setup_logging, save_model

logger = setup_logging("fraud_model")

def train_baseline_rf(X_train, y_train, random_state=42):
    """
    Trains a default Random Forest model (baseline).
    """
    logger.info("Training baseline RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    logger.info("Baseline training complete!")
    return rf

def train_weighted_rf(X_train, y_train, random_state=42):
    """
    Trains a Random Forest model with class_weight='balanced'.
    """
    logger.info("Training class-weighted RandomForestClassifier (class_weight='balanced')...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    logger.info("Class-weighted training complete!")
    return rf

def train_smote_rf(X_train, y_train, random_state=42):
    """
    Applies SMOTE to balance the dataset, then trains a Random Forest.
    """
    logger.info("Applying SMOTE to balance training dataset...")
    smote = SMOTE(random_state=random_state, sampling_strategy=0.1)  # Upsample minority to 10% of majority to be safe and memory efficient
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"Resampled training set size: {X_train_res.shape[0]:,} (Fraud cases: {y_train_res.sum():,})")
    
    logger.info("Training RandomForestClassifier on SMOTE resampled data...")
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)
    logger.info("SMOTE training complete!")
    return rf

def tune_hyperparameters(X_train, y_train, random_state=42):
    """
    Tuning hyperparameters of Random Forest using GridSearchCV.
    Uses a small grid and sample size to run quickly.
    """
    logger.info("Tuning hyperparameters using GridSearchCV...")
    
    # Stratified downsample if too large, to run fast in sandboxed environment
    if len(X_train) > 50000:
        logger.info("Downsampling training set to 20,000 rows for hyperparameter tuning...")
        # Separate classes
        fraud_idx = y_train[y_train == 1].index
        non_fraud_idx = y_train[y_train == 0].index
        
        # Sample non-fraud
        sampled_non_fraud = np.random.choice(non_fraud_idx, size=18000, replace=False)
        # Keep all fraud or sample if too many
        sampled_fraud = np.random.choice(fraud_idx, size=min(2000, len(fraud_idx)), replace=False)
        
        sample_indices = np.concatenate([sampled_non_fraud, sampled_fraud])
        X_sample = X_train.loc[sample_indices]
        y_sample = y_train.loc[sample_indices]
    else:
        X_sample = X_train
        y_sample = y_train
        
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Focus tuning on recall and F1 due to class imbalance
    rf = RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_sample, y_sample)
    logger.info(f"GridSearchCV complete! Best F1 Score: {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def generate_fraud_scores(model, X):
    """
    Generates fraud probability scores between 0 and 100.
    """
    logger.info("Generating fraud probability scores...")
    probs = model.predict_proba(X)[:, 1]
    scores = np.round(probs * 100, 2)
    return scores

def assign_risk_category(scores):
    """
    Assigns risk category based on fraud score:
    - Low Risk: 0-30
    - Medium Risk: 30-70
    - High Risk: 70-100
    """
    categories = []
    for s in scores:
        if s < 30.0:
            categories.append('Low Risk')
        elif s < 70.0:
            categories.append('Medium Risk')
        else:
            categories.append('High Risk')
    return np.array(categories)
