import numpy as np
from sklearn.metrics import precision_recall_curve, auc

def find_optimal_threshold(y_true, y_probs, min_precision=0.2):
    """
    Cari threshold terbaik berdasarkan F1-score dan batas minimum precision.
    
    Args:
        y_true: array-like of shape (n_samples,) — label asli
        y_probs: array-like of shape (n_samples,) — probabilitas prediksi
        min_precision: float — precision minimum yang diizinkan

    Returns:
        dict: {
            'best_threshold': float,
            'best_precision': float,
            'best_recall': float,
            'best_f1': float,
            'pr_auc': float
        }
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    pr_auc = auc(recalls, precisions)

    # Filter berdasarkan precision minimum
    valid = precisions >= min_precision
    if not np.any(valid):
        return {
            'best_threshold': 0.5,
            'best_precision': 0.0,
            'best_recall': 0.0,
            'best_f1': 0.0,
            'pr_auc': pr_auc
        }

    best_idx = np.argmax(f1_scores * valid)

    return {
        'best_threshold': thresholds[best_idx],
        'best_precision': precisions[best_idx],
        'best_recall': recalls[best_idx],
        'best_f1': f1_scores[best_idx],
        'pr_auc': pr_auc
    }
