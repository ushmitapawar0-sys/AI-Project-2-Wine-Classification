"""
CLI script to make predictions using the trained model.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from src.config import MODELS_DIR

def main():
    parser = argparse.ArgumentParser(description="Predict wine class from 13 chemical features.")
    parser.add_argument('features', metavar='F', type=float, nargs=13,
                        help='13 feature values separated by spaces')
    
    args = parser.parse_args()
    
    # Load model
    model_path = MODELS_DIR / 'best_model.joblib'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Please run 'python -m src.train' first.")
        return

    model = joblib.load(model_path)
    
    # Feature names
    data = load_wine()
    feature_names = data.feature_names
    class_names = data.target_names
    
    # Create DataFrame for prediction (preserves feature names for scaler)
    X_new = pd.DataFrame([args.features], columns=feature_names)
    
    # Predict
    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0] if hasattr(model, 'predict_proba') else None
    
    print("\n--- Prediction Results ---")
    print(f"Predicted Class: {class_names[prediction]} (Class {prediction})")
    
    if probabilities is not None:
        print("\nClass Probabilities:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {probabilities[i]:.4f}")

if __name__ == "__main__":
    main()
