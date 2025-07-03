import numpy as np

def majority_voting(preds):
    # preds: list of array-like (n_models, n_samples)
    preds = np.array(preds)
    votes = np.sum(preds, axis=0)
    # Majority voting: positive if >= 2 votes
    return (votes >= 2).astype(int)
