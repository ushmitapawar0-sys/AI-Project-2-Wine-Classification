"""
Main training pipeline script.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

from src.config import RANDOM_STATE, TEST_SIZE, TARGET_ACCURACY, MODELS_DIR, REPORTS_DIR
from src.data import load_wine_data
from src.preprocess import create_preprocessor
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_feature_importance

def get_models():
    """Returns a dictionary of models to evaluate."""
    return {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=RANDOM_STATE),
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE)
    }

def main():
    print("1. Loading dataset...")
    X, y = load_wine_data()
    data = load_wine()
    class_names = data.target_names
    feature_names = data.feature_names
    
    print(f"2. Splitting data (Test size: {TEST_SIZE}, Stratified)...")
    # Stratify ensures the class distribution is identical in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print("3. Comparing initial models using Cross Validation (5-fold)...")
    models = get_models()
    preprocessor = create_preprocessor()
    
    results = []
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        results.append({'Model': name, 'CV_Accuracy_Mean': cv_scores.mean(), 'CV_Accuracy_Std': cv_scores.std()})
    
    results_df = pd.DataFrame(results).sort_values(by='CV_Accuracy_Mean', ascending=False)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    
    # Pick top 2 models for tuning
    top_2_names = results_df['Model'].head(2).tolist()
    print(f"\n4. Tuning top 2 models: {top_2_names}")
    
    param_grids = {
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10]
        },
        'Logistic Regression': {
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['lbfgs', 'liblinear']
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        },
        'KNN': {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance']
        },
        'Decision Tree': {
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5]
        }
    }
    
    best_overall_model = None
    best_overall_score = 0
    best_overall_name = ""
    
    for name in top_2_names:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', models[name])
        ])
        grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        # We tune only on the training set. Touching the test set here causes data leakage!
        grid_search.fit(X_train, y_train)
        
        print(f"   Best params for {name}: {grid_search.best_params_} (CV Score: {grid_search.best_score_:.4f})")
        
        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_overall_model = grid_search.best_estimator_
            best_overall_name = name
            
    print(f"\n5. Best overall model chosen: {best_overall_name}")
    
    print("6. Evaluating best model on untouched test set...")
    y_pred = best_overall_model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n7. Generating and saving plots...")
    plot_confusion_matrix(y_test, y_pred, classes=class_names)
    
    # Extract actual classifier step from pipeline to plot feature importance
    classifier_step = best_overall_model.named_steps['classifier']
    plot_feature_importance(classifier_step, feature_names)
    
    print("\n8. Decision Loop...")
    if metrics['accuracy'] < TARGET_ACCURACY:
        print(f"WARNING: Test accuracy {metrics['accuracy']:.4f} is below target {TARGET_ACCURACY}.")
        # In a real workflow we might loop back here.
    else:
        print(f"SUCCESS: Test accuracy {metrics['accuracy']:.4f} meets target {TARGET_ACCURACY}.")
    
    print("\n9. Saving model and metrics...")
    joblib.dump(best_overall_model, MODELS_DIR / 'best_model.joblib')
    
    # Save results to markdown
    with open(REPORTS_DIR / 'results.md', 'w') as f:
        f.write("# Wine Classification Metrics Summary\n\n")
        f.write("## Model Comparison\n")
        f.write(results_df.to_markdown(index=False))
        f.write(f"\n\n## Best Model: {best_overall_name}\n\n")
        f.write("### Test Set Evaluation\n")
        for k, v in metrics.items():
            f.write(f"- **{k.capitalize()}**: {v:.4f}\n")

    print("Done! Pipeline execution complete.")

if __name__ == "__main__":
    main()
