
from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "netflix_churn.csv"
MODEL_PATH = BASE_DIR / "Models" / "churn_model.pkl"

# Load Netflix dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

if "churned" not in df.columns:
    raise ValueError("Dataset mein 'churned' column nahi mila.")

# Prepare target
y = (
    df["churned"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

if y.isna().any():
    raise ValueError("'churned' column mein sirf Yes/No values honi chahiye.")

# Prepare input features
X = df.drop(columns=["churned", "subscriber_id"], errors="ignore")

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("Netflix Churn Prediction Model")
print("--------------------------------")
print("Total subscribers:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Stay", "Churn"],
    zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Save trained model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print(f"\nModel saved successfully: {MODEL_PATH}")
