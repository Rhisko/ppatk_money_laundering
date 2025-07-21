import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.threshold_tuning import find_optimal_threshold
import pandas as pd

# def evaluate_model(name, model, X_test_scaled, y_test):
#     y_pred = model.predict(X_test_scaled)
#     print(f"=== {name.upper()} ===")
#     print(f"Akurasi   : {accuracy_score(y_test, y_pred):.4f}")
#     print(classification_report(y_test, y_pred, digits=4))
#     print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
#     print("="*40)

def evaluate_model(name, model, X_test_scaled, y_test, threshold=0.5, auto_tune=False):
    default_threshold = threshold
    if hasattr(model, "predict_proba"):
        # Gunakan probabilitas dan threshold custom
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        if auto_tune:
            result = find_optimal_threshold(y_test, y_proba, min_precision=0.2)
            threshold = result["best_threshold"]
            if threshold <= 0.0:
                print(f"[AUTO TUNE] {name} => Threshold tidak valid, menggunakan default 0.5")
                threshold = default_threshold
            print(f"[AUTO TUNE] {name} => Threshold={threshold:.4f} | Precision={result['best_precision']:.4f} | Recall={result['best_recall']:.4f} | F1={result['best_f1']:.4f} | PR-AUC={result['pr_auc']:.4f}")
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




