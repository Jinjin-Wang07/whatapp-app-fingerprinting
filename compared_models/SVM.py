import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

def svm_clf(X_train, Y_train, X_test, Y_test):
    np.random.seed(42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Apply the same transformation to the test set
    
    # SVM cross-validation parameter grid
    param_grid = {
        'C': [4],
        'gamma': [0.125],
        'kernel': ['rbf'],
        'decision_function_shape': ['ovo']  # Suitable for multi-class classification
    }
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    svm = SVC()
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, Y_train)
    
    # # Output best parameters and cross-validation accuracy
    # print("Best Parameters:", grid_search.best_params_)
    # print("CrossValidation Accuracy for best Parameters:", grid_search.best_score_)
    
    # Train the final model with the best parameters
    best_svm = grid_search.best_estimator_
    best_svm.fit(X_train_scaled, Y_train)
    
    # Make predictions on the test set
    test_preds = best_svm.predict(X_test_scaled)
    test_acc = accuracy_score(Y_test, test_preds)
    print(f"Test Accuracy: {test_acc}")

    # Compute and print classification report
    report = classification_report(Y_test, test_preds, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(Y_test, test_preds))  # Print the full report

    print(f"\nTest Accuracy: {report['accuracy']:.4f}")
    # # Extract and print F1-score
    # if "weighted avg" in report:
    #     f1_score = report["weighted avg"]["f1-score"]
    #     print(f"\nWeighted F1-score: {f1_score}")

    return report