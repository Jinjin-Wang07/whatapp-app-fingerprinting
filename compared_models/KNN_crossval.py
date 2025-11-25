from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def knn_clf(X_train, Y_train, X_test, Y_test, weights="distance"):
    """
    Training a KNN model and evaluate the classification efficiency with test data

    Args:
    - X_train: Dataframe of size N x M where N is the number of samples and M is the number of features
    - Y_train: DataSeries of size N
    """
    
    # Create Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(weights=weights))
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        "knn__n_neighbors": [5] #range(1, 20)  # Test n_neighbors from 1 to 30
    }
    
    # Perform GridSearchCV with 10-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Prediction
    y_pred = best_model.predict(X_test)
    
    # # Evaluation
    # print("Confusion Matrix:")
    # print(confusion_matrix(Y_test, y_pred))

    rp = classification_report(Y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred))

    print(f"\nTest Accuracy: {rp['accuracy']:.4f}")
    
    return rp