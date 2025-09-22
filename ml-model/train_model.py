# train_model.py
from src.train import train_model

if __name__ == "__main__":
    model = train_model("data/equipment_data.csv")
