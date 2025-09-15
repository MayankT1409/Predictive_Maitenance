from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    model, X_test, y_test = train_model("data/equipment_data.csv")  # ❌ remove model_type
    evaluate_model(model, X_test, y_test)
