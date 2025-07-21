from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_model(y_train):
    # Dapatkan kelas unik dan hitung bobot otomatis
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # Buat dictionary class_weight seperti {0: ..., 1: ...}
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    return RandomForestClassifier(
        class_weight=class_weight_dict,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
