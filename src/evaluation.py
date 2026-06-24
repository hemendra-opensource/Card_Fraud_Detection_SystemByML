import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.utils import setup_logging, ensure_dir

logger = setup_logging("evaluation")

def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    Computes and logs classification metrics.
    Returns a dictionary of metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['roc_auc'] = 0.0
        
    logger.info("--- Classification Metrics ---")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
    if y_prob is not None:
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plots the confusion matrix using Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved confusion matrix plot to {save_path}")
    plt.close()
    return cm

def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plots the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved ROC curve plot to {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, save_path=None, top_n=20):
    """
    Extracts, ranks, and plots feature importances from a Random Forest model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': [importances[i] for i in indices]
    })
    
    # Filter top_n
    feat_imp_top = feat_imp.head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_top, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Value')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved feature importance plot to {save_path}")
    plt.close()
    
    return feat_imp

def evaluate_cross_validation(model, X, y, cv=5, random_state=42):
    """
    Performs K-Fold Cross Validation and reports mean metrics.
    """
    logger.info(f"Running {cv}-Fold Stratified Cross Validation...")
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_results = cross_validate(
        model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1, return_train_score=False
    )
    
    summary = {}
    logger.info(f"\n--- {cv}-Fold CV Results Summary ---")
    for metric in scoring:
        key = f"test_{metric}"
        scores = cv_results[key]
        summary[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        logger.info(f"Mean {metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
    return summary
