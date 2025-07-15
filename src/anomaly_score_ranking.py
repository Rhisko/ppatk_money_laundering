import pandas as pd
import numpy as np

def rank_anomaly_scores(y_true, scores, top_n=100, sort_desc=True):
    """
    Mengurutkan skor anomaly dan menampilkan Top-N transaksi paling mencurigakan.

    Args:
        y_true: array-like, ground truth label (0: normal, 1: fraud)
        scores: array-like, skor anomaly hasil hybrid/soft voting
        top_n: int, jumlah kasus teratas yang ingin ditampilkan
        sort_desc: bool, apakah urut dari skor tertinggi

    Returns:
        DataFrame berisi:
            - index
            - true label
            - skor anomaly
            - rank
    """
    df = pd.DataFrame({
        'label': y_true,
        'score': scores
    })

    df['rank'] = df['score'].rank(method='first', ascending=not sort_desc)
    df_sorted = df.sort_values(by='score', ascending=not sort_desc).head(top_n)

    return df_sorted.reset_index(drop=True)
