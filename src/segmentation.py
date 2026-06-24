import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import setup_logging, save_model

logger = setup_logging("segmentation")

def aggregate_customer_data(df, scores):
    """
    Aggregates transaction-level data into customer-level profiles.
    Parameters:
      df: Transaction DataFrame (raw or filtered)
      scores: NumPy array of fraud probability scores (0-100)
    """
    logger.info("Aggregating transaction-level data into customer profiles...")
    
    # Add score and suspicious indicator to dataframe
    df_temp = df.copy()
    df_temp['fraud_score'] = scores
    df_temp['is_suspicious'] = (scores >= 30.0).astype(int)
    
    # Aggregation
    customer_agg = df_temp.groupby('nameOrig').agg(
        transaction_frequency=('amount', 'count'),
        average_transaction_amount=('amount', 'mean'),
        total_spending=('amount', 'sum'),
        fraud_probability_score=('fraud_score', 'mean'),
        number_of_suspicious_transactions=('is_suspicious', 'sum'),
        fraud_count=('isFraud', 'sum'),
        average_balance=('oldbalanceOrg', 'mean')
    ).reset_index()
    
    logger.info(f"Aggregated profiles for {customer_agg.shape[0]:,} unique customers.")
    return customer_agg

def normalize_customer_features(customer_df, scaler_path='models/scaler_kmeans.pkl', is_training=True):
    """
    Normalizes customer-level features for clustering using StandardScaler.
    """
    logger.info(f"Normalizing customer features (is_training={is_training})...")
    feature_cols = [
        'transaction_frequency', 'average_transaction_amount', 'total_spending',
        'fraud_probability_score', 'number_of_suspicious_transactions', 'fraud_count', 'average_balance'
    ]
    
    X = customer_df[feature_cols].copy()
    
    # Fill any NaNs
    X = X.fillna(0)
    
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        save_model(scaler, scaler_path)
    else:
        from src.utils import load_model
        scaler = load_model(scaler_path)
        X_scaled = scaler.transform(X)
        
    return X_scaled, feature_cols

def run_elbow_method(X_scaled, max_k=10):
    """
    Computes KMeans inertia (WCSSE) for k=1 to max_k to help determine optimal clusters.
    """
    logger.info("Running Elbow Method to determine optimal K...")
    inertias = []
    ks = list(range(1, max_k + 1))
    
    # We sample if dataset is too large to make it fast
    if len(X_scaled) > 50000:
        indices = np.random.choice(len(X_scaled), size=50000, replace=False)
        X_sample = X_scaled[indices]
    else:
        X_sample = X_scaled
        
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_sample)
        inertias.append(kmeans.inertia_)
        
    return ks, inertias

def train_kmeans(X_scaled, k=4, model_path='models/kmeans.pkl'):
    """
    Trains the final K-Means model on the scaled features.
    """
    logger.info(f"Training K-Means with K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    save_model(kmeans, model_path)
    logger.info("K-Means training complete!")
    return kmeans

def apply_pca(X_scaled, n_components=2):
    """
    Applies Principal Component Analysis (PCA) to reduce features to 2D for visualization.
    """
    logger.info("Applying PCA for cluster visualization...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"Explained variance ratio by component: {pca.explained_variance_ratio_}")
    return X_pca, pca

def map_cluster_labels(customer_df, kmeans_model, feature_cols):
    """
    Maps cluster integer labels to meaningful business segment labels based on centroids.
    Segments:
      1. Low-Risk Regular Customers
      2. High-Value Customers
      3. High-Risk Customers
      4. Fraud-Prone Customers
      5. Dormant Customers
    """
    logger.info("Analyzing clusters to assign business names...")
    
    # Assign cluster labels to df
    customer_df = customer_df.copy()
    
    # Calculate cluster averages
    centroids = customer_df.groupby('cluster')[feature_cols].mean()
    print("Cluster Centroids:")
    print(centroids)
    
    cluster_labels = {}
    
    # Heuristics to auto-label based on centroids
    # 1. Fraud-Prone: High average fraud_count / fraud_probability
    # 2. High-Risk: High suspicious transactions / medium fraud
    # 3. High-Value: High average_balance / total_spending
    # 4. Dormant: Low transaction frequency and low spending
    # 5. Low-Risk Regular: Moderate/Low frequency, zero/near-zero fraud count
    
    for cluster_id in centroids.index:
        row = centroids.loc[cluster_id]
        
        if row['fraud_count'] > 0.5 or row['fraud_probability_score'] > 50:
            cluster_labels[cluster_id] = "Fraud-Prone Customers"
        elif row['number_of_suspicious_transactions'] > 0.5 or row['fraud_probability_score'] > 20:
            cluster_labels[cluster_id] = "High-Risk Customers"
        elif row['total_spending'] > customer_df['total_spending'].mean() * 1.5 or row['average_balance'] > customer_df['average_balance'].mean() * 1.5:
            cluster_labels[cluster_id] = "High-Value Customers"
        elif row['transaction_frequency'] <= 1.2 and row['total_spending'] < customer_df['total_spending'].mean() * 0.5:
            cluster_labels[cluster_id] = "Dormant Customers"
        else:
            cluster_labels[cluster_id] = "Low-Risk Regular Customers"
            
    # Resolve any duplicate labels by adding a number or adjusting
    # For example, if two clusters map to Low-Risk Regular, we can keep them or sub-label them.
    used_labels = {}
    final_labels = {}
    for cid, label in cluster_labels.items():
        if label not in used_labels:
            used_labels[label] = 1
            final_labels[cid] = label
        else:
            used_labels[label] += 1
            final_labels[cid] = f"{label} (Group {used_labels[label]})"
            
    customer_df['segment_label'] = customer_df['cluster'].map(final_labels)
    logger.info(f"Mapped cluster labels: {final_labels}")
    return customer_df, final_labels
