import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import setup_logging, save_model, ensure_dir
from src.data_preprocessing import load_raw_data, clean_data, filter_fraud_susceptible, split_data_stratified
from src.feature_engineering import preprocess_features
from src.fraud_model import (
    train_baseline_rf, train_weighted_rf, train_smote_rf, 
    tune_hyperparameters, generate_fraud_scores, assign_risk_category
)
from src.segmentation import (
    aggregate_customer_data, normalize_customer_features, 
    run_elbow_method, train_kmeans, apply_pca, map_cluster_labels
)
from src.evaluation import (
    evaluate_classification, plot_confusion_matrix, 
    plot_roc_curve, plot_feature_importance, evaluate_cross_validation
)

logger = setup_logging("train_pipeline")

def run_pipeline():
    logger.info("Starting End-to-End Machine Learning Pipeline...")
    
    # 1. Load Data
    raw_path = "data/raw/PS_20174392719_1491204439457_log.csv"
    df = load_raw_data(raw_path)
    
    # 2. Clean Data
    df = clean_data(df)
    
    # 3. Filter for Susceptible Transactions (TRANSFER & CASH_OUT)
    df_filtered = filter_fraud_susceptible(df)
    
    # 4. Stratified Split (80-20)
    X_train_raw, X_test_raw, y_train, y_test = split_data_stratified(df_filtered)
    
    # To run quickly and prevent memory exhaustion, downsample training set
    # while preserving ALL fraud cases
    max_train_size = 200000
    if len(X_train_raw) > max_train_size:
        logger.info(f"Downsampling training set to {max_train_size:,} rows for efficiency while keeping ALL fraud cases...")
        
        # Get indices
        fraud_idx = y_train[y_train == 1].index
        non_fraud_idx = y_train[y_train == 0].index
        
        # We sample non-fraud to fill up the budget
        num_non_fraud_sample = max_train_size - len(fraud_idx)
        sampled_non_fraud = np.random.choice(non_fraud_idx, size=num_non_fraud_sample, replace=False)
        
        # Concatenate indices
        train_sample_idx = np.concatenate([fraud_idx, sampled_non_fraud])
        
        # Subset training set
        X_train_raw = X_train_raw.loc[train_sample_idx]
        y_train = y_train.loc[train_sample_idx]
        
        logger.info(f"New training set size: {len(X_train_raw):,} (Fraud cases: {y_train.sum():,})")
        
    # We will also downsample the test set slightly for speed, but keep it representative
    max_test_size = 50000
    if len(X_test_raw) > max_test_size:
        logger.info(f"Downsampling test set to {max_test_size:,} rows for evaluation speed...")
        fraud_idx = y_test[y_test == 1].index
        non_fraud_idx = y_test[y_test == 0].index
        
        num_non_fraud_sample = max_test_size - len(fraud_idx)
        sampled_non_fraud = np.random.choice(non_fraud_idx, size=num_non_fraud_sample, replace=False)
        
        test_sample_idx = np.concatenate([fraud_idx, sampled_non_fraud])
        X_test_raw = X_test_raw.loc[test_sample_idx]
        y_test = y_test.loc[test_sample_idx]
        
        logger.info(f"New test set size: {len(X_test_raw):,} (Fraud cases: {y_test.sum():,})")
        
    # Reassemble train and test df for preprocessing (since we need original columns for feature engineering)
    train_df = X_train_raw.copy()
    train_df['isFraud'] = y_train
    
    test_df = X_test_raw.copy()
    test_df['isFraud'] = y_test
    
    # 5. Preprocess & Feature Engineer (fits scaler on train, applies to test)
    scaler_path = "models/scaler.pkl"
    X_train = preprocess_features(train_df, is_training=True, scaler_path=scaler_path)
    X_test = preprocess_features(test_df, is_training=False, scaler_path=scaler_path)
    
    # 6. Train Baseline Model
    baseline_model = train_baseline_rf(X_train, y_train)
    logger.info("Evaluating Baseline Model on Test Set:")
    y_pred_base = baseline_model.predict(X_test)
    y_prob_base = baseline_model.predict_proba(X_test)[:, 1]
    metrics_base = evaluate_classification(y_test, y_pred_base, y_prob_base)
    
    # 7. Compare Imbalance Handling (Weighted RF vs SMOTE RF)
    weighted_model = train_weighted_rf(X_train, y_train)
    logger.info("Evaluating Class-Weighted Model on Test Set:")
    y_pred_weighted = weighted_model.predict(X_test)
    y_prob_weighted = weighted_model.predict_proba(X_test)[:, 1]
    metrics_weighted = evaluate_classification(y_test, y_pred_weighted, y_prob_weighted)
    
    smote_model = train_smote_rf(X_train, y_train)
    logger.info("Evaluating SMOTE Model on Test Set:")
    y_pred_smote = smote_model.predict(X_test)
    y_prob_smote = smote_model.predict_proba(X_test)[:, 1]
    metrics_smote = evaluate_classification(y_test, y_pred_smote, y_prob_smote)
    
    # Explain which model is better
    logger.info("\n--- Imbalance Handling Method Comparison ---")
    logger.info(f"Baseline F1: {metrics_base['f1_score']:.4f} | Recall: {metrics_base['recall']:.4f} | Precision: {metrics_base['precision']:.4f}")
    logger.info(f"Weighted F1: {metrics_weighted['f1_score']:.4f} | Recall: {metrics_weighted['recall']:.4f} | Precision: {metrics_weighted['precision']:.4f}")
    logger.info(f"SMOTE F1:    {metrics_smote['f1_score']:.4f} | Recall: {metrics_smote['recall']:.4f} | Precision: {metrics_smote['precision']:.4f}")
    
    # Determine best method (typically Class Weighted or SMOTE depending on F1)
    # We choose the Class-Weighted model as it usually maintains better precision than SMOTE on PaySim
    best_initial_model = weighted_model if metrics_weighted['f1_score'] >= metrics_smote['f1_score'] else smote_model
    logger.info(f"Choosing best initial approach for tuning: {'Class-Weighted' if best_initial_model == weighted_model else 'SMOTE'}")
    
    # 8. Hyperparameter Tuning
    best_tuned_model, best_params, best_score = tune_hyperparameters(X_train, y_train)
    
    # Train final optimized model on full training set
    logger.info("Training final optimized model on training data...")
    final_model = best_tuned_model
    final_model.fit(X_train, y_train)
    
    # Save final model
    rf_path = "models/random_forest.pkl"
    save_model(final_model, rf_path)
    
    # 9. Evaluate Final Model
    logger.info("Evaluating Final Tuned Model on Test Set:")
    y_pred_final = final_model.predict(X_test)
    y_prob_final = final_model.predict_proba(X_test)[:, 1]
    metrics_final = evaluate_classification(y_test, y_pred_final, y_prob_final)
    
    # Save evaluation plots
    fig_dir = "reports/figures"
    ensure_dir(fig_dir)
    
    plot_confusion_matrix(y_test, y_pred_final, save_path=os.path.join(fig_dir, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_prob_final, save_path=os.path.join(fig_dir, "roc_curve.png"))
    
    # 10. Feature Importance Analysis
    feature_names = list(X_train.columns)
    feat_imp = plot_feature_importance(final_model, feature_names, save_path=os.path.join(fig_dir, "feature_importance.png"))
    print("\nTop 10 Features:")
    print(feat_imp.head(10))
    
    # 11. 5-Fold Cross Validation
    cv_summary = evaluate_cross_validation(final_model, X_train, y_train, cv=5)
    
    # 12. Fraud Probability Scoring (over full filtered dataset to save processed csv)
    logger.info("Generating fraud scores for the entire dataset...")
    # Preprocess full filtered dataset (without fitting scaler)
    X_full = preprocess_features(df_filtered, is_training=False, scaler_path=scaler_path)
    full_scores = generate_fraud_scores(final_model, X_full)
    full_risk_categories = assign_risk_category(full_scores)
    
    # Save processed transaction data
    df_scored = df_filtered.copy()
    df_scored['fraud_score'] = full_scores
    df_scored['risk_category'] = full_risk_categories
    
    scored_path = "data/processed/scored_transactions.csv"
    ensure_dir(os.path.dirname(scored_path))
    # For size efficiency, we can save a subset of scored transactions or full
    df_scored.to_csv(scored_path, index=False)
    logger.info(f"Saved scored transactions dataset to {scored_path}")
    
    # 13. Customer Segmentation (Clustering)
    customer_df = aggregate_customer_data(df_filtered, full_scores)
    
    # Normalize features
    scaler_kmeans_path = "models/scaler_kmeans.pkl"
    X_cust_scaled, cust_features = normalize_customer_features(customer_df, scaler_kmeans_path, is_training=True)
    
    # Run Elbow Method
    ks, WCSSE = run_elbow_method(X_cust_scaled, max_k=8)
    
    # Plot Elbow Curve
    plt.figure(figsize=(7, 4))
    plt.plot(ks, WCSSE, 'bo-')
    plt.title('Elbow Curve for K-Means')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSSE)')
    plt.tight_layout()
    elbow_path = os.path.join(fig_dir, "elbow_curve.png")
    plt.savefig(elbow_path, dpi=300)
    plt.close()
    logger.info(f"Saved Elbow curve to {elbow_path}")
    
    # Train K-Means (Optimal K based on elbows is typically 4 or 5)
    optimal_k = 4
    kmeans_model = train_kmeans(X_cust_scaled, k=optimal_k)
    
    # Predict clusters
    customer_df['cluster'] = kmeans_model.predict(X_cust_scaled)
    
    # Map business labels
    customer_df, label_mapping = map_cluster_labels(customer_df, kmeans_model, cust_features)
    
    # Apply PCA for 2D visualization
    X_pca, pca = apply_pca(X_cust_scaled)
    customer_df['pca1'] = X_pca[:, 0]
    customer_df['pca2'] = X_pca[:, 1]
    
    # Downsample for visualization to prevent slow plotting and memory overhead
    logger.info("Downsampling customer profiles to 50,000 for plotting...")
    plot_df = customer_df.sample(n=min(50000, len(customer_df)), random_state=42)
    
    # Save PCA scatter plot
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=plot_df, x='pca1', y='pca2', hue='segment_label', 
        palette='Set2', style='segment_label', alpha=0.8
    )
    plt.title('Customer Segments in PCA 2D Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    pca_plot_path = os.path.join(fig_dir, "pca_clusters.png")
    plt.savefig(pca_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved PCA cluster plot to {pca_plot_path}")
    
    # Save customer segments dataset
    segments_path = "data/processed/customer_segments.csv"
    customer_df.to_csv(segments_path, index=False)
    logger.info(f"Saved customer segments dataset to {segments_path}")
    
    logger.info("Pipeline Execution Complete! All models and assets have been saved.")

if __name__ == "__main__":
    run_pipeline()
