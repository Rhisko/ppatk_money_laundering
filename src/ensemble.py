import numpy as np

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

def hybrid_voting(prob_preds, threshold=0.3, weights=None):
    """
    Hybrid voting antara soft voting dan majority threshold voting.
    
    Args:
        prob_preds: List of array-like, shape (n_models, n_samples)
        threshold: Batas probabilitas agar dianggap positif
        weights: Optional list of weights untuk soft voting

    Returns:
        Array of binary predictions (1 = positif, 0 = negatif)
    """
    prob_preds = np.array(prob_preds)  # shape: (n_models, n_samples)

    # Soft voting: probabilitas rata-rata
    if weights:
        weights = np.array(weights)
        avg_probs = np.average(prob_preds, axis=0, weights=weights)
    else:
        avg_probs = np.mean(prob_preds, axis=0)
    
    # Majority voting: berdasarkan apakah masing-masing model ≥ threshold
    binary_preds = (prob_preds >= threshold).astype(int)
    majority_votes = np.sum(binary_preds, axis=0)
    
    # Final prediksi: positif jika soft voting positif ATAU majority voting positif
    final_preds = np.logical_or(
        avg_probs >= threshold,
        majority_votes >= 2
    ).astype(int)
    
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
