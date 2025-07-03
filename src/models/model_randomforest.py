from sklearn.ensemble import RandomForestClassifier

def get_model():
    return RandomForestClassifier(n_jobs=-1, class_weight='balanced')
