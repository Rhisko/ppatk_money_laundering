import numpy as np
from src.threshold_tuning import find_optimal_threshold

def majority_voting(preds):
    # preds: list of array-like (n_models, n_samples)
    preds = np.array(preds)
    votes = np.sum(preds, axis=0)
    # Majority voting: positive if >= 2 votes
    return (votes >= 2).astype(int)

def weighted_majority_voting(preds, weights):
    preds = np.array(preds)
    weights = np.array(weights).reshape(-1, 1)
    weighted_votes = np.sum(preds * weights, axis=0)
    return (weighted_votes >= np.sum(weights) / 2).astype(int)

# def hybrid_voting(prob_preds, threshold=0.3, weights=None):
#     """
#     Hybrid voting antara soft voting dan majority threshold voting.
    
#     Args:
#         prob_preds: List of array-like, shape (n_models, n_samples)
#         threshold: Batas probabilitas agar dianggap positif
#         weights: Optional list of weights untuk soft voting

#     Returns:
#         Array of binary predictions (1 = positif, 0 = negatif)
#     """
#     prob_preds = np.array(prob_preds)  # shape: (n_models, n_samples)

#     # Soft voting: probabilitas rata-rata
#     if weights:
#         weights = np.array(weights)
#         avg_probs = np.average(prob_preds, axis=0, weights=weights)
#     else:
#         avg_probs = np.mean(prob_preds, axis=0)
    
#     # Majority voting: berdasarkan apakah masing-masing model ≥ threshold
#     binary_preds = (prob_preds >= threshold).astype(int)
#     majority_votes = np.sum(binary_preds, axis=0)
    
#     # Final prediksi: positif jika soft voting positif ATAU majority voting positif
#     final_preds = np.logical_or(
#         avg_probs >= threshold,
#         majority_votes >= 2
#     ).astype(int)
    
#     return final_preds

def hybrid_voting(prob_preds, threshold=0.5, weights=None, soft_weight=0.7, hard_weight=0.3, return_score=False, auto_tune=False):
    """
    Hybrid voting adaptif: kombinasi soft voting dan majority voting (binary) dengan bobot fleksibel.

    Args:
        prob_preds: List of array-like, shape (n_models, n_samples)
        threshold: Threshold akhir untuk menentukan klasifikasi positif
        weights: Optional list of weights untuk soft voting antar model
        soft_weight: Bobot kontribusi soft voting (0-1)
        hard_weight: Bobot kontribusi majority voting (0-1)
        return_score: Jika True, mengembalikan skor akhir anomaly

    Returns:
        Array of binary predictions (1 = positif, 0 = negatif)
        Optionally: Array of final hybrid score (float)
    """
    prob_preds = np.array(prob_preds)  # (n_models, n_samples)

    # Soft voting score (weighted or mean)
    if weights:
        weights = np.array(weights)
        soft_scores = np.average(prob_preds, axis=0, weights=weights)
    else:
        soft_scores = np.mean(prob_preds, axis=0)

    # Hard voting score (jumlah model yang memprediksi positif)
    binary_preds = (prob_preds >= 0.5).astype(int)
    majority_score = np.sum(binary_preds, axis=0) / prob_preds.shape[0]  # normalize to 0-1

    # Hybrid score: kombinasi dua pendekatan
    final_scores = soft_weight * soft_scores + hard_weight * majority_score

    # Final binary decision
    final_preds = (final_scores >= threshold).astype(int)

    if return_score:
        return final_preds, final_scores
    return final_preds


def majority_soft_voting(prob_preds, threshold=0.3):
    """
    Melakukan majority voting berdasarkan probabilitas masing-masing model.
    Suatu prediksi dianggap positif jika ≥2 model memiliki probabilitas ≥ threshold.
    
    Args:
        prob_preds: List of array-like, shape (n_models, n_samples)
        threshold: Threshold probabilitas untuk menganggap suatu prediksi sebagai positif
    
    Returns:
        Array of binary predictions (1=positif, 0=negatif)
    """
    prob_preds = np.array(prob_preds)  # shape: (n_models, n_samples)
    # Binerkan prediksi tiap model berdasarkan threshold
    binary_preds = (prob_preds >= threshold).astype(int)
    # Hitung jumlah model yang memprediksi positif untuk tiap sample
    votes = np.sum(binary_preds, axis=0)
    # Prediksi final: positif jika ≥ 2 dari 3 model memprediksi positif
    return (votes >= 2).astype(int)
