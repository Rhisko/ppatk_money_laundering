from src.preprocessing import load_data, preprocess_data , split_and_scale
from src.train import train_all_models
from src.testing import evaluate_model
from imblearn.over_sampling import SMOTE
import inquirer
from inquirer.themes import GreenPassion
import joblib
from config.config import DATA_PATH , MODELS

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
        loaded_models[name] = joblib.load(f"trained_models/{name}_model_20250614_v1.pkl")

    for name, model in loaded_models.items():
        evaluate_model(name, model, X_test, y_test)

    # # Predict with each model
    # preds = []
    # for model in models.values():
    #     preds.append(model.predict(X_test))

    # # Evaluate individual models
    # print("Individual Model Evaluation:")
    # for name, model in models.items():
    #     y_pred = model.predict(X_test)
    #     print(f"{name}: {evaluate(y_test, y_pred)}")

    # # Ensemble prediction
    # ensemble_pred = majority_voting(preds)
    # print("Ensemble Evaluation:")
    # print(evaluate(y_test, ensemble_pred))

    # # Save models (optional)
    # os.makedirs('models', exist_ok=True)
    # for name, model in models.items():
    #     joblib.dump(model, f'models/{name}.pkl')

if __name__ == "__main__":
    main()
