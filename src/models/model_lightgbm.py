import lightgbm as lgb

def get_model():
    return lgb.LGBMClassifier(n_jobs=-1, class_weight='balanced')
