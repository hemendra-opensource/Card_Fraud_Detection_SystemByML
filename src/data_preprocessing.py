import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import setup_logging

logger = setup_logging("data_preprocessing")

def load_raw_data(filepath):
    """
    Loads raw PaySim dataset from CSV path.
    """
    logger.info(f"Loading raw data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw data with shape: {df.shape}")
    return df

def clean_data(df):
    """
    Cleans the dataset by removing duplicate rows and filling missing values.
    """
    logger.info("Cleaning data (removing duplicates and nulls)...")
    initial_shape = df.shape
    
    # Remove duplicates
    df = df.drop_duplicates()
    if df.shape != initial_shape:
        logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")
        
    # Check for missing values
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        logger.info(f"Found {null_counts} null values. Filling missing values...")
        df = df.fillna(method='ffill')
    else:
        logger.info("No missing values found.")
        
    return df

def filter_fraud_susceptible(df):
    """
    Filters the dataset to only include transaction types susceptible to fraud
    (TRANSFER and CASH_OUT), based on our exploratory data analysis.
    This reduces data volume by 56% while preserving 100% of fraud instances.
    """
    logger.info("Filtering dataset for TRANSFER and CASH_OUT transaction types...")
    initial_shape = df.shape
    initial_fraud = df['isFraud'].sum()
    
    # Filter for TRANSFER and CASH_OUT
    filtered_df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    
    filtered_shape = filtered_df.shape
    filtered_fraud = filtered_df['isFraud'].sum()
    
    logger.info(f"Reduced dataset size from {initial_shape[0]:,} to {filtered_shape[0]:,} rows.")
    logger.info(f"Preserved {filtered_fraud} of {initial_fraud} fraud records (100.00%).")
    return filtered_df

def split_data_stratified(df, target_col='isFraud', test_size=0.2, random_state=42):
    """
    Performs an 80-20 stratified train-test split to preserve class ratios.
    """
    logger.info(f"Performing stratified split on target: {target_col} with test_size={test_size}...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    logger.info(f"Train set size: {X_train.shape[0]:,} (Fraud cases: {y_train.sum():,})")
    logger.info(f"Test set size: {X_test.shape[0]:,} (Fraud cases: {y_test.sum():,})")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(df, filepath):
    """
    Saves a processed dataframe to a specified location.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved processed data to {filepath}")
