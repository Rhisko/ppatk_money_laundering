import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

def evaluate_model(name, model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    print(f"=== {name.upper()} ===")
    print(f"Akurasi   : {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("="*40)




