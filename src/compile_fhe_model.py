import pandas as pd
from sklearn.model_selection import train_test_split
import concrete.ml as cml
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.deployment import FHEModelDev
import os
import numpy as np
import joblib

# Configuration
DATA_FILE = "data/synthetic_health_data.csv"
FHE_MODEL_DIR = "models/fhe_model"
TEST_SIZE = 0.2 # Must match the split in train_plaintext_model.py
RANDOM_STATE = 42 # Must match the split in train_plaintext_model.py

# We'll create a clean directory for the FHE model
# If the directory exists, we'll remove its contents first
if os.path.exists(FHE_MODEL_DIR):
    print(f"Cleaning existing directory: {FHE_MODEL_DIR}")
    import shutil
    # Remove all files in the directory
    for file_name in os.listdir(FHE_MODEL_DIR):
        file_path = os.path.join(FHE_MODEL_DIR, file_name)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
else:
    # Create the directory if it doesn't exist
    os.makedirs(FHE_MODEL_DIR)

def compile_fhe():
    """Loads data, fits a Concrete-ML model, compiles to FHE, and saves."""
    print(f"Loading data from {DATA_FILE} for training and calibration...")
    df = pd.read_csv(DATA_FILE)
    X = df.drop('Risk', axis=1)
    y = df['Risk']

    # Need X_train, y_train for fitting, and X_train for calibration
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Instantiating Concrete-ML Logistic Regression model...")
    concrete_model = ConcreteLogisticRegression()

    # Convert X_train to numpy array for fitting/compilation
    # Cast to float32 as required by Concrete-ML quantization
    X_train_np = X_train.to_numpy().astype(np.float32)
    y_train_np = y_train.to_numpy() # Also need y_train for fitting

    print("Fitting the Concrete-ML model...")
    # Fit the model using the training data
    concrete_model.fit(X_train_np, y_train_np)

    print("Compiling the fitted model to FHE...")
    # Compile the model using the calibration data (X_train_np)
    # Compile modifies the concrete_model object in place and returns the circuit
    _ = concrete_model.compile(X_train_np) # We don't need the returned circuit directly

    # Save the compiled model using FHEModelDev
    print("Saving the fitted and compiled Concrete ML model using FHEModelDev...")
    try:
        # Create an FHEModelDev instance with our model
        dev = FHEModelDev(path_dir=FHE_MODEL_DIR, model=concrete_model)

        # Save both client.zip and server.zip
        dev.save()
        print("FHE model components saved successfully using FHEModelDev.")

        # Verify files were created
        server_zip_path = os.path.join(FHE_MODEL_DIR, "server.zip")
        client_zip_path = os.path.join(FHE_MODEL_DIR, "client.zip")

        if os.path.exists(server_zip_path) and os.path.exists(client_zip_path):
            print("Successfully verified server.zip and client.zip creation.")
            print(f"Server components saved to: {server_zip_path}")
            print(f"Client components saved to: {client_zip_path}")
        else:
            print("Warning: server.zip or client.zip not found after saving attempt.")
            if not os.path.exists(server_zip_path):
                print(f"Missing: {server_zip_path}")
            if not os.path.exists(client_zip_path):
                print(f"Missing: {client_zip_path}")

    except Exception as e:
        print(f"Error during model save: {e}")
        print("Saving failed. This might be due to internal serialization issues.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compile_fhe()