from src.ensemble import weighted_majority_voting,majority_voting,hybrid_voting,majority_soft_voting
from src.evaluate import evaluate
from src.preprocessing import load_data, preprocess_data , split_and_scale
from src.train import train_all_models
from src.testing import evaluate_model
from imblearn.over_sampling import SMOTE
import inquirer
from inquirer.themes import GreenPassion
import joblib
from config.config import DATA_PATH , MODELS
import datetime

# Public variable to store model thresholds
MODEL_THRESHOLDS = {
    "randomforest": 0.3,
    "xgboost": 0.7,
    "lightgbm": 0.3,
}
weights = [0.3, 0.7, 0.3]  # Example weights for each model
def get_user_input(prompt_message, choices):
    question = [inquirer.List('choice', message=prompt_message, choices=choices)]
    answer = inquirer.prompt(question, theme=GreenPassion())
    return answer["choice"]

def main():
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
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        loaded_models[name] = joblib.load(f"trained_models/{name}_model_{today_str}_v1.pkl")

    preds = []
    for name, model in loaded_models.items():
        print(f"Evaluating model: {name}")
        threshold = MODEL_THRESHOLDS.get(name, 0.5)
        eval = evaluate_model(name, model, X_test, y_test, threshold=threshold, plot_pr=False)
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
    res_majority_voting = majority_voting(preds)
    print("Ensemble Evaluation - Majority:")
    print(evaluate(y_test, res_majority_voting))

    res_weighted_majority_voting = weighted_majority_voting(preds,weights)
    print("Ensemble Evaluation - Weighted Majority:")
    print(evaluate(y_test, res_weighted_majority_voting))

    res_majority_soft_voting = majority_soft_voting(preds,0.5)
    print("Ensemble Evaluation - Majority Soft Voting:")
    print(evaluate(y_test, res_majority_soft_voting))
    
    res_hybrid_voting = hybrid_voting(preds,0.5, weights)
    print("Ensemble Evaluation - Hybrid Voting:")
    print(evaluate(y_test, res_hybrid_voting))

    # # Save models (optional)
    # os.makedirs('models', exist_ok=True)
    # for name, model in models.items():
    #     joblib.dump(model, f'models/{name}.pkl')

if __name__ == "__main__":
    main()
