import numpy as np
import pandas as pd
import os

# Configuration
N_SAMPLES = 500  # Number of synthetic data points to generate
OUTPUT_DIR = "data"
OUTPUT_FILE = "synthetic_health_data.csv"

# Define feature ranges (adjust as needed for realistic values)
AGE_RANGE = (25, 75)
SYSTOLIC_BP_RANGE = (90, 180)
DIASTOLIC_BP_RANGE = (60, 110)
BLOOD_SUGAR_RANGE = (70, 150)  # mg/dL

# New feature ranges
BMI_RANGE = (18.5, 40.0)  # Normal to Obese
TOTAL_CHOLESTEROL_RANGE = (120, 300)  # mg/dL
HDL_CHOLESTEROL_RANGE = (30, 90)  # mg/dL
LDL_CHOLESTEROL_RANGE = (50, 200)  # mg/dL
# Binary features (0 or 1)
SMOKING_PROB = 0.25  # 25% chance of being a smoker
FAMILY_HISTORY_PROB = 0.30  # 30% chance of family history of heart disease

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_data(n_samples):
    """Generates synthetic health data."""
    # Generate random features within specified ranges
    age = np.random.randint(AGE_RANGE[0], AGE_RANGE[1], n_samples)
    systolic_bp = np.random.randint(SYSTOLIC_BP_RANGE[0], SYSTOLIC_BP_RANGE[1], n_samples)
    diastolic_bp = np.random.randint(DIASTOLIC_BP_RANGE[0], DIASTOLIC_BP_RANGE[1], n_samples)
    blood_sugar = np.random.randint(BLOOD_SUGAR_RANGE[0], BLOOD_SUGAR_RANGE[1], n_samples)

    # Generate new features
    bmi = np.random.uniform(BMI_RANGE[0], BMI_RANGE[1], n_samples).round(1)  # Round to 1 decimal place
    total_cholesterol = np.random.randint(TOTAL_CHOLESTEROL_RANGE[0], TOTAL_CHOLESTEROL_RANGE[1], n_samples)
    hdl_cholesterol = np.random.randint(HDL_CHOLESTEROL_RANGE[0], HDL_CHOLESTEROL_RANGE[1], n_samples)
    ldl_cholesterol = np.random.randint(LDL_CHOLESTEROL_RANGE[0], LDL_CHOLESTEROL_RANGE[1], n_samples)

    # Binary features
    smoking = (np.random.random(n_samples) < SMOKING_PROB).astype(int)
    family_history = (np.random.random(n_samples) < FAMILY_HISTORY_PROB).astype(int)

    # Create a more comprehensive risk score based on thresholds
    # (This is a placeholder - a real model would be trained on real data)
    risk_score = (
        (age > 55) * 0.15 +                   # Age is a risk factor
        (systolic_bp > 140) * 0.15 +           # High systolic BP
        (diastolic_bp > 90) * 0.10 +           # High diastolic BP
        (blood_sugar > 125) * 0.10 +           # High blood sugar
        (bmi > 30) * 0.10 +                    # Obesity (BMI > 30)
        (total_cholesterol > 240) * 0.10 +     # High total cholesterol
        (hdl_cholesterol < 40) * 0.10 +        # Low HDL ("good") cholesterol
        (ldl_cholesterol > 160) * 0.10 +       # High LDL ("bad") cholesterol
        smoking * 0.15 +                       # Smoking is a major risk factor
        family_history * 0.10                  # Family history of heart disease
    )

    # Add some noise
    risk_score += np.random.normal(0, 0.05, n_samples)

    # Determine binary risk outcome based on a threshold on the score
    # Adjust threshold to balance classes if needed
    risk_threshold = 0.5
    risk_outcome = (risk_score > risk_threshold).astype(int) # 0 = Low Risk, 1 = High Risk

    # Create DataFrame with all features
    df = pd.DataFrame({
        'Age': age,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp,
        'Blood_Sugar': blood_sugar,
        'BMI': bmi,
        'Total_Cholesterol': total_cholesterol,
        'HDL_Cholesterol': hdl_cholesterol,
        'LDL_Cholesterol': ldl_cholesterol,
        'Smoking': smoking,
        'Family_History': family_history,
        'Risk': risk_outcome
    })

    return df

if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} synthetic health records...")
    synthetic_data = generate_data(N_SAMPLES)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    synthetic_data.to_csv(output_path, index=False)

    print(f"Data saved to {output_path}")
    print("First 5 rows:")
    print(synthetic_data.head())
    print(f"\nValue counts for Risk: {synthetic_data['Risk'].value_counts()}")