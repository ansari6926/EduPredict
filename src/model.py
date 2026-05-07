from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import os

def train_model(df):
    """
    Trains a Random Forest Classifier to predict student grades.
    Returns the trained model, accuracy, and feature importances.
    """
    features = ['Attendance', 'Internal1', 'Internal2', 'Assignment', 'Participation']
    X = df[features]
    y = df['Grade']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    importances = dict(zip(features, model.feature_importances_.round(3)))
    
    return model, accuracy, importances

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    try:
        df = pd.read_csv("../dataset/raw_data/student_data.csv")
        model, acc, imp = train_model(df)
        print(f"Model trained successfully. Accuracy: {acc*100:.2f}%")
        print("Feature Importances:", imp)
        save_model(model, "../outputs/results/model.pkl")
    except FileNotFoundError:
        print("Data not found. Please run preprocessing.py first.")
