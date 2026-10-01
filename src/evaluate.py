"""
Module for evaluating model performance.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.config import FIGURES_DIR
import numpy as np
import pandas as pd
import json

def plot_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix.png"):
    """
    Plots and saves the confusion matrix as a heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def plot_feature_importance(model, feature_names, filename="feature_importance.png"):
    """
    Plots feature importance if the model supports it.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not support feature importance plotting directly.")
        return
        
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), sorted_features, rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def evaluate_model(y_true, y_pred) -> dict:
    """
    Computes key classification metrics.
    
    Returns:
        Dictionary containing accuracy, precision, recall, and f1.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics
