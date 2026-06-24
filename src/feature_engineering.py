import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils import setup_logging, save_model, load_model

logger = setup_logging("feature_engineering")

def create_features(df):
    """
    Creates high-value domain features from raw transactions.
    Works for both a full DataFrame and a single row dictionary (as DataFrame).
    """
    # Prevent SettingWithCopyWarning
    df = df.copy()
    
    # 1. balance_difference_orig (ideal behavior: old - new - amount == 0)
    # Fraud transactions often deplete accounts to 0, leaving old - new == amount, so difference is 0.
    # Safe transactions might fail or be bounded, leaving differences.
    df['balance_difference_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    
    # 2. balance_difference_dest (ideal behavior: new - old - amount == 0)
    # In fraud, newbalanceDest is often NOT updated (remains 0) even though transfer happened,
    # causing a high negative difference.
    df['balance_difference_dest'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']
    
    # 3. transaction_amount_ratio (Ratio of transaction amount to sender's starting balance)
    # Fraud transactions often try to transfer the entire balance, making this ratio close to 1.
    df['transaction_amount_ratio'] = np.where(
        df['oldbalanceOrg'] > 0, 
        df['amount'] / df['oldbalanceOrg'], 
        df['amount']  # fallback if old balance is 0
    )
    
    # 4. account_risk_indicator
    # Indicator if receiver starts with 'M' (merchant) or 'C' (customer)
    df['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
    
    # 5. transaction_velocity
    # Number of transactions by this sender in the current step (hour)
    # If it is a single record, velocity is defaulted to 1.
    if len(df) > 1:
        velocity_map = df.groupby(['nameOrig', 'step']).size().to_dict()
        df['transaction_velocity'] = df.apply(lambda row: velocity_map.get((row['nameOrig'], row['step']), 1), axis=1)
    else:
        df['transaction_velocity'] = 1
        
    return df

def preprocess_features(df, is_training=True, scaler_path='models/scaler.pkl'):
    """
    Applies feature engineering, encoding, and scaling.
    If is_training is True, it fits and saves the scaler.
    If is_training is False, it loads and applies the saved scaler.
    """
    logger.info(f"Preprocessing features (is_training={is_training})...")
    
    # 1. Create features
    df_feat = create_features(df)
    
    # 2. One-hot encoding for transaction type (TRANSFER vs CASH_OUT)
    # Even if only one type is present (e.g. inference), we ensure both columns exist.
    if 'type_TRANSFER' not in df_feat.columns:
        df_feat['type_TRANSFER'] = (df_feat['type'] == 'TRANSFER').astype(int)
    if 'type_CASH_OUT' not in df_feat.columns:
        df_feat['type_CASH_OUT'] = (df_feat['type'] == 'CASH_OUT').astype(int)
        
    # 3. Define feature columns
    feature_cols = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'balance_difference_orig', 
        'balance_difference_dest', 'transaction_amount_ratio', 
        'dest_is_merchant', 'transaction_velocity',
        'type_TRANSFER', 'type_CASH_OUT'
    ]
    
    X = df_feat[feature_cols].copy()
    
    # Fill any NaNs created by ratios or calculations
    X = X.fillna(0)
    
    # 4. Scaling numeric columns
    numeric_cols = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'balance_difference_orig', 
        'balance_difference_dest', 'transaction_amount_ratio', 'transaction_velocity'
    ]
    
    if is_training:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        # Save scaler
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        save_model(scaler, scaler_path)
        logger.info(f"Fitted and saved scaler to {scaler_path}")
    else:
        scaler = load_model(scaler_path)
        X[numeric_cols] = scaler.transform(X[numeric_cols])
        logger.info(f"Applied loaded scaler from {scaler_path}")
        
    return X
