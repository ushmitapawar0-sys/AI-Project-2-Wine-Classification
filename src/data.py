"""
Module to load and prepare the Wine dataset.
"""
from sklearn.datasets import load_wine
import pandas as pd
from typing import Tuple

def load_wine_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the Wine dataset from scikit-learn.
    
    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target labels.
    """
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y
