from xgboost import XGBClassifier

def get_model(y_train):
    weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    return XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state=42, scale_pos_weight=weight_ratio)
