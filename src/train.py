import time
from src.models import model_xgboost, model_lightgbm, model_randomforest
import joblib
import datetime

def train_all_models(X_train, y_train, verbose=True):
    models = {
        "xgboost": model_xgboost.get_model(y_train),
        "lightgbm": model_lightgbm.get_model(y_train),
        "randomforest": model_randomforest.get_model(y_train)
    }
    train_times = {}
    for name, model in models.items():
        if verbose:
            print(f"Training {name} ...")
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        train_times[name] = end - start
        if verbose:
            print(f"{name} completed in {end - start:.2f} seconds\n")
            # Save the trained model
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            joblib.dump(model, f"trained_models/{name}_model_{current_date}_v1.pkl")
            
            
    return models, train_times
