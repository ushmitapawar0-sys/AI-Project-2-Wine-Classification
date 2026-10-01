"""
Module for data preprocessing.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

def create_preprocessor() -> Pipeline:
    """
    Creates a scikit-learn Pipeline for data preprocessing.
    Currently only uses StandardScaler. By putting it in a Pipeline,
    we ensure there is no data leakage from test set to training set.
    
    Returns:
        Pipeline: Preprocessing pipeline.
    """
    # Scaling is crucial for distance-based algorithms like KNN and SVM
    return Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
