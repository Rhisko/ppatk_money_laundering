from src.ensemble import weighted_majority_voting,majority_voting,hybrid_voting,majority_soft_voting
from src.evaluate import evaluate
from src.anomaly_score_ranking import rank_anomaly_scores
from src.preprocessing import load_data, preprocess_data , split_and_scale
from src.train import train_all_models
from src.testing import evaluate_model
from imblearn.over_sampling import SMOTE
import inquirer
from inquirer.themes import GreenPassion
import joblib
from config.config import DATA_PATH , MODELS
import datetime
import pandas as pd

# Public variable to store model thresholds
MODEL_THRESHOLDS = {
    "randomforest": 0.3,
    "xgboost": 0.7,
    "lightgbm": 0.6,
}
weights = [0.3, 0.5, 0.3]  # Example weights for each model
def get_user_input(prompt_message, choices):
    question = [inquirer.List('choice', message=prompt_message, choices=choices)]
    answer = inquirer.prompt(question, theme=GreenPassion())
    return answer["choice"]

def main():
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    process_step = get_user_input("Please choose the step in the money laundering detection process:", 
            [
            "Model_Training",
            "Model_Testing",
            "Model_Evaluation",
            "Model_Voting",]
        )
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)


    # sm = SMOTE(random_state=42)
    # X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    # Train individual models
    if process_step == "Model_Training":
        models, train_times = train_all_models(X_train, y_train, verbose=True)
        print("Training completed. Models and training times:")
        for name, model in models.items():
            print(f"{name}: {model}")
        print("Training times:", train_times)


    # evaluate individual models
    loaded_models = {}
    for name in MODELS:
        
        loaded_models[name] = joblib.load(f"trained_models/{name}_model_{today_str}_v1.pkl")

    preds = []
    for name, model in loaded_models.items():
        print(f"Evaluating model: {name}")
        threshold = MODEL_THRESHOLDS.get(name, 0.5)
        eval = evaluate_model(name, model, X_test, y_test, threshold=threshold, auto_tune=True)
        preds.append(eval)

    # Predict with each model
    # for name, model in loaded_models.items():
    #     preds.append(model.predict(X_test))

    # Evaluate individual models
    # print("Individual Model Evaluation:")
    # for name, model in loaded_models.items():
    #     y_pred = model.predict(X_test)
    #     print(f"{name}: {evaluate(y_test, y_pred)}")

    # Ensemble prediction
    # res_majority_voting = majority_voting(preds)
    # print("Ensemble Evaluation - Majority:")
    # print(evaluate(y_test, res_majority_voting))

    # res_weighted_majority_voting = weighted_majority_voting(preds,weights)
    # print("Ensemble Evaluation - Weighted Majority:")
    # print(evaluate(y_test, res_weighted_majority_voting))

    # res_majority_soft_voting = majority_soft_voting(preds,0.5)
    # print("Ensemble Evaluation - Majority Soft Voting:")
    # print(evaluate(y_test, res_majority_soft_voting))
    
    res_hybrid_voting, hybrid_scores = hybrid_voting(preds, threshold=0.3, weights=weights, return_score=True, auto_tune=True)
    print("Ensemble Evaluation - Hybrid Voting (Adaptive):")
    print(f"Hybrid Scores: {hybrid_scores}")
    df_scores = pd.DataFrame({
        "label": y_test,
        "hybrid_score": hybrid_scores
    })
    df_scores.to_csv("hybrid_anomaly_scores.csv", index=False)
    print(evaluate(y_test, res_hybrid_voting))

    top_suspicious = rank_anomaly_scores(y_test, hybrid_scores, top_n=100)

    print("Top 10 suspicious transactions (ranked by hybrid anomaly score):")
    print(top_suspicious.head(10))

    # Simpan ke file
    top_suspicious.to_csv(f"data/results/top_suspicious_transactions_{today_str}.csv", index=False)

    # # Save models (optional)
    # os.makedirs('models', exist_ok=True)
    # for name, model in models.items():
    #     joblib.dump(model, f'models/{name}.pkl')

if __name__ == "__main__":
    main()
