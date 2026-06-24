import os
import pandas as pd
import numpy as np

def inspect_dataset():
    file_path = os.path.abspath("data/raw/PS_20174392719_1491204439457_log.csv")
    print(f"Reading dataset from {file_path}...")
    
    # Read first 10000 rows to quickly get schema info (or read full info for simple statistics)
    # The dataset has ~6.3 million rows, which is ~470MB. Let's load the full dataset to get accurate metrics.
    df = pd.read_csv(file_path)
    
    print("\n--- Dataset Overview ---")
    print(f"Shape of dataset: {df.shape}")
    print(f"Number of rows: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\n--- Columns and Data Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Duplicate Records ---")
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    
    print("\n--- Sample Records ---")
    print(df.head(5))
    
    print("\n--- Target Class Imbalance (isFraud) ---")
    fraud_counts = df['isFraud'].value_counts()
    fraud_pct = df['isFraud'].value_counts(normalize=True) * 100
    for val, count in fraud_counts.items():
        print(f"Class {val}: {count:,} records ({fraud_pct[val]:.4f}%)")
        
    print("\n--- Target Class Imbalance (isFlaggedFraud) ---")
    flagged_counts = df['isFlaggedFraud'].value_counts()
    flagged_pct = df['isFlaggedFraud'].value_counts(normalize=True) * 100
    for val, count in flagged_counts.items():
        print(f"Class {val}: {count:,} records ({flagged_pct[val]:.4f}%)")
        
    print("\n--- Transaction Types ---")
    type_counts = df['type'].value_counts()
    type_pct = df['type'].value_counts(normalize=True) * 100
    for val, count in type_counts.items():
        print(f"Type '{val}': {count:,} records ({type_pct[val]:.2f}%)")
        
    print("\n--- Fraud by Transaction Type ---")
    fraud_by_type = df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean'])
    fraud_by_type['mean'] = fraud_by_type['mean'] * 100
    fraud_by_type.columns = ['Total Transactions', 'Fraud Count', 'Fraud Rate (%)']
    print(fraud_by_type)
    
    print("\n--- Numerical Columns Describe ---")
    num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    print(df[num_cols].describe())
    
    print("\n--- Unique Senders and Receivers ---")
    print(f"Unique senders (nameOrig): {df['nameOrig'].nunique():,}")
    print(f"Unique receivers (nameDest): {df['nameDest'].nunique():,}")
    
if __name__ == "__main__":
    inspect_dataset()
