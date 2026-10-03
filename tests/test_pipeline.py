import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from src.data import load_wine_data
from src.preprocess import create_preprocessor
from src.config import TARGET_ACCURACY
from src.evaluate import evaluate_model
import numpy as np

def test_data_loads():
    """Test if the dataset loads correctly and has expected shapes."""
    X, y = load_wine_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == 178
    assert X.shape[1] == 13
    assert len(y) == 178

def test_pipeline_fits():
    """Test if the preprocessing pipeline fits without error."""
    X, y = load_wine_data()
    preprocessor = create_preprocessor()
    assert isinstance(preprocessor, Pipeline)
    
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape == X.shape
    
    # Check if data is standardized (mean approx 0, std approx 1)
    assert np.allclose(X_transformed.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_transformed.std(axis=0), 1, atol=1e-7)

def test_prediction_shape():
    """Mock prediction to check shape output."""
    from sklearn.ensemble import RandomForestClassifier
    X, y = load_wine_data()
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Predict on one sample
    pred = model.predict(X.iloc[[0]])
    assert len(pred) == 1
    assert pred[0] in [0, 1, 2]

def test_accuracy_above_threshold():
    """Check if a basic model achieves the minimum target accuracy."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    X, y = load_wine_data()
    preprocessor = create_preprocessor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)
    
    metrics = evaluate_model(y_test, y_pred)
    assert metrics['accuracy'] >= TARGET_ACCURACY
