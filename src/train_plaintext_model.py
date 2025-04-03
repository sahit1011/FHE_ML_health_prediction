import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Configuration
DATA_FILE = "data/synthetic_health_data.csv"
MODEL_DIR = "models"
MODEL_FILE = "plaintext_model.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    """Loads data, trains a plaintext model, evaluates, and saves it."""
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # Separate features (X) and target (y)
    X = df.drop('Risk', axis=1)
    y = df['Risk']

    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    print("Training Logistic Regression model...")
    # For simplicity and FHE compatibility, we use a basic Logistic Regression
    # No scaling is applied here, but Concrete-ML handles quantization internally.
    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    print("Evaluating model on test data...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Plaintext model accuracy: {accuracy:.4f}")

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)

    print("Model training complete.")
    return X_train, X_test # Return data splits for potential use in FHE compilation

if __name__ == "__main__":
    train_model()