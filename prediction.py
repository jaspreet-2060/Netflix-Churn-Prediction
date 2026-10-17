
from pathlib import Path
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


def predict_churn(customer_data):
    # Convert customer details to a DataFrame
    data = pd.DataFrame([customer_data])

    # Remove ID if provided; model was trained without it
    data = data.drop(columns=["subscriber_id"], errors="ignore")

    # Predict churn probability
    probabilities = model.predict_proba(data)[0]
    classes = list(model.classes_)

    churn_index = classes.index(1)
    stay_index = classes.index(0)

    churn_probability = float(probabilities[churn_index])
    stay_probability = float(probabilities[stay_index])

    if churn_probability >= 0.5:
        result = "Subscriber is likely to CHURN"
    else:
        result = "Subscriber is likely to STAY"

    return result, churn_probability, stay_probability
