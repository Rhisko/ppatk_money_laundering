import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# def evaluate_model(name, model, X_test_scaled, y_test):
#     y_pred = model.predict(X_test_scaled)
#     print(f"=== {name.upper()} ===")
#     print(f"Akurasi   : {accuracy_score(y_test, y_pred):.4f}")
#     print(classification_report(y_test, y_pred, digits=4))
#     print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
#     print("="*40)

def evaluate_model(name, model, X_test_scaled, y_test, threshold=0.5, plot_pr=False):
    if hasattr(model, "predict_proba"):
        # Gunakan probabilitas dan threshold custom
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        # Model tidak punya predict_proba (jarang), pakai prediksi langsung
        y_pred = model.predict(X_test_scaled)

    print(f"=== {name.upper()} @ Threshold={threshold:.2f} ===")
    print(f"Akurasi   : {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("="*40)
    return y_pred




