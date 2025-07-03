from xgboost import XGBClassifier

def get_model():
    return XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state=42)
