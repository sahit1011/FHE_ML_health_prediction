# src/test_inference.py

import joblib
import numpy as np
import os
import time

# Import our client and server classes
from fhe_client import FHEClient
from fhe_prediction_service import FHEServer

# Configuration
PLAINTEXT_MODEL_FILE = "models/plaintext_model.joblib"
FHE_MODEL_DIR = "models/fhe_model"

# Sample input data (ensure dtype is float32 for FHE consistency)
# Features: Age, Systolic_BP, Diastolic_BP, Blood_Sugar, BMI, Total_Cholesterol, HDL_Cholesterol, LDL_Cholesterol, Smoking, Family_History

# Low risk sample: Younger person with good health metrics
sample_data_low_risk = np.array([
    [40.0, 120.0, 80.0, 90.0, 22.5, 180.0, 60.0, 100.0, 0.0, 0.0]
], dtype=np.float32)

# High risk sample: Older person with concerning health metrics
sample_data_high_risk = np.array([
    [65.0, 150.0, 95.0, 140.0, 32.0, 250.0, 35.0, 180.0, 1.0, 1.0]
], dtype=np.float32)

# Borderline case: Mixed risk factors
sample_data_borderline = np.array([
    [55.0, 135.0, 88.0, 120.0, 28.0, 210.0, 45.0, 140.0, 0.0, 1.0]
], dtype=np.float32)

def test_single_prediction(sample_data, sample_name):
    """Tests both plaintext and FHE prediction for a single data sample."""
    print(f"--- Testing Prediction for: {sample_name} ---")

    # Print input data in a more readable format
    print("Input data:")
    feature_names = ["Age", "Systolic BP", "Diastolic BP", "Blood Sugar",
                    "BMI", "Total Cholesterol", "HDL Cholesterol", "LDL Cholesterol",
                    "Smoking", "Family History"]

    for i, feature in enumerate(feature_names):
        value = sample_data[0][i]
        # Format binary features as Yes/No
        if feature in ["Smoking", "Family History"]:
            value_str = "Yes" if value == 1.0 else "No"
        else:
            value_str = f"{value:.1f}"
        print(f"  {feature}: {value_str}")

    # --- 1. Plaintext Prediction ---
    print("\n[Plaintext Workflow]")
    if not os.path.exists(PLAINTEXT_MODEL_FILE):
        print(f"Error: Plaintext model not found at {PLAINTEXT_MODEL_FILE}")
        return

    plaintext_model = joblib.load(PLAINTEXT_MODEL_FILE)
    print(f"Plaintext model loaded from {PLAINTEXT_MODEL_FILE}")

    start_time_plain = time.time()
    # Scikit-learn expects DataFrame or 2D numpy array
    plaintext_prediction = plaintext_model.predict(sample_data)
    plaintext_proba = plaintext_model.predict_proba(sample_data) # Get probabilities too
    end_time_plain = time.time()

    print(f"Plaintext Predicted Class: {plaintext_prediction[0]}")
    print(f"Plaintext Predicted Probabilities: {plaintext_proba[0]}")
    print(f"Plaintext Inference Time: {end_time_plain - start_time_plain:.4f} seconds")

    # --- 2. FHE Prediction ---
    print("\n[FHE Workflow]")
    try:
        # Initialize Client and Server
        fhe_client = FHEClient(model_dir=FHE_MODEL_DIR)
        # The server loads the model during initialization
        fhe_server = FHEServer(model_dir=FHE_MODEL_DIR)

        # Client encrypts data
        start_time_enc = time.time()
        encrypted_input = fhe_client.encrypt_data(sample_data)
        end_time_enc = time.time()
        print(f"Encryption Time: {end_time_enc - start_time_enc:.4f} seconds")
        print(f"Encrypted input type: {type(encrypted_input)}") # Show type, not the data itself

        # Get evaluation keys from client
        evaluation_keys = fhe_client.get_evaluation_keys()

        # Server performs FHE prediction
        start_time_fhe_pred = time.time()
        encrypted_prediction = fhe_server.predict(encrypted_input, evaluation_keys)
        end_time_fhe_pred = time.time()
        print(f"FHE Inference Time (on server): {end_time_fhe_pred - start_time_fhe_pred:.4f} seconds")

        # Client decrypts result
        start_time_dec = time.time()
        decrypted_prediction = fhe_client.decrypt_result(encrypted_prediction)
        end_time_dec = time.time()
        print(f"Decryption Time: {end_time_dec - start_time_dec:.4f} seconds")

        # Note: FHE models often return the predicted class directly.
        # Getting probabilities might require different model compilation settings.
        print(f"Decrypted FHE Predicted Class: {decrypted_prediction[0]}") # Access the value inside the array

        # --- Comparison ---
        print("\n[Comparison]")

        # Get the predicted class from FHE probabilities (index of max value)
        fhe_class = np.argmax(decrypted_prediction[0])

        # Format the risk labels
        plaintext_risk = "High Risk" if plaintext_prediction[0] == 1 else "Low Risk"
        fhe_risk = "High Risk" if fhe_class == 1 else "Low Risk"

        # Calculate probability differences
        prob_diff = np.abs(plaintext_proba[0] - decrypted_prediction[0]) * 100

        print(f"Plaintext Model Prediction: {plaintext_risk} (Class {plaintext_prediction[0]})")
        print(f"FHE Model Prediction: {fhe_risk} (Class {fhe_class})")
        print("\nProbability Comparison:")
        print(f"  Plaintext Low Risk: {plaintext_proba[0][0]:.4f} ({plaintext_proba[0][0]*100:.1f}%)")
        print(f"  FHE Low Risk:      {decrypted_prediction[0][0]:.4f} ({decrypted_prediction[0][0]*100:.1f}%)")
        print(f"  Difference:        {prob_diff[0]:.2f}%")
        print(f"\n  Plaintext High Risk: {plaintext_proba[0][1]:.4f} ({plaintext_proba[0][1]*100:.1f}%)")
        print(f"  FHE High Risk:      {decrypted_prediction[0][1]:.4f} ({decrypted_prediction[0][1]*100:.1f}%)")
        print(f"  Difference:         {prob_diff[1]:.2f}%")

        # Calculate average probability difference
        avg_diff = np.mean(prob_diff)

        print(f"\nAverage Probability Difference: {avg_diff:.2f}%")

        if plaintext_prediction[0] == fhe_class:
            print("\nResult: ✅ MATCH - Both models predict the same outcome")
        else:
            print("\nResult: ❌ MISMATCH - Models predict different outcomes")
            print("  (This can occur due to quantization in the FHE model)")

        # Provide an assessment of the similarity
        if avg_diff < 1.0:
            print("\nSimilarity Assessment: Excellent (< 1% difference)")
        elif avg_diff < 5.0:
            print("\nSimilarity Assessment: Good (< 5% difference)")
        elif avg_diff < 10.0:
            print("\nSimilarity Assessment: Fair (< 10% difference)")
        else:
            print("\nSimilarity Assessment: Poor (> 10% difference)")

    except FileNotFoundError as e:
        print(f"Error loading FHE components: {e}")
        print("Ensure both client.zip and server.zip exist in the FHE model directory.")
    except Exception as e:
        print(f"An error occurred during the FHE workflow: {e}")

    print("-" * (len(sample_name) + 28))


def print_summary_header():
    """Prints a formatted header for the summary section."""
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY OF RESULTS")
    print("="*80)

def print_summary_row(sample_name, plaintext_result, fhe_result, match_status, similarity):
    """Prints a formatted row for the summary table."""
    print(f"{sample_name:<20} | {plaintext_result:<15} | {fhe_result:<15} | {match_status:<10} | {similarity}")

if __name__ == "__main__":
    # Store results for summary
    results = []

    # Test low risk sample
    print("\n" + "#"*80)
    print(" "*30 + "TEST CASE 1: LOW RISK")
    print("#"*80)
    test_single_prediction(sample_data_low_risk, "Low Risk Sample")

    # Test high risk sample
    print("\n" + "#"*80)
    print(" "*30 + "TEST CASE 2: HIGH RISK")
    print("#"*80)
    test_single_prediction(sample_data_high_risk, "High Risk Sample")

    # Test borderline case
    print("\n" + "#"*80)
    print(" "*30 + "TEST CASE 3: BORDERLINE CASE")
    print("#"*80)
    test_single_prediction(sample_data_borderline, "Borderline Case")
