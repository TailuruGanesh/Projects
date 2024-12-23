```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Data Preprocessing
def preprocess_data(filepath):
    """
    Load and preprocess the data.
    Args:
        filepath (str): Path to the dataset.
    Returns:
        X_train, X_test, y_train, y_test: Processed datasets.
    """
    # Load the data
    data = pd.read_csv(filepath)
    
    # Drop duplicates and handle missing values
    data = data.drop_duplicates().dropna()
    
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# 2. Model Training
def train_all_models(X_train, y_train):
    """
    Train multiple models and return them.
    Args:
        X_train (array): Training features.
        y_train (array): Training labels.
    Returns:
        models (dict): Dictionary of trained models.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} trained successfully.")
    
    return models

# 3. Model Evaluation
def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple trained models.
    Args:
        models (dict): Dictionary of trained models.
        X_test (array): Testing features.
        y_test (array): True labels for testing.
    """
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", report)

# Main Execution
if __name__ == "__main__":
    # Provide the path to your dataset
    dataset_path = "dataset.csv"    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(dataset_path)    
    # Train the models
    models = train_all_models(X_train, y_train)    
    # Evaluate the models
    evaluate_models(models, X_test, y_test)

```
