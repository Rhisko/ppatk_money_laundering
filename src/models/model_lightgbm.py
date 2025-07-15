import lightgbm as lgb

def get_model(y_train):
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    return lgb.LGBMClassifier(
        scale_pos_weight=ratio,
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
