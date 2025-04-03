# Secure Health Prediction using FHE-ML

![FHE Health Prediction](static/images/fhe_health_prediction.png)

A privacy-preserving health risk prediction system using Fully Homomorphic Encryption (FHE) and Machine Learning.

## Project Overview

**Goal:** This project demonstrates how Fully Homomorphic Encryption (FHE) can be used to perform Machine Learning predictions on sensitive health data *without* exposing the raw data to the server performing the prediction.

**Problem:** Users want to leverage ML-powered health services (e.g., risk prediction) but are concerned about the privacy of their highly sensitive medical information. Sending unencrypted data poses significant privacy risks.

**Solution using FHE:**
FHE allows computations (like ML model inference) to be performed directly on encrypted data.
1. The user encrypts their health metrics using their personal key.
2. The encrypted data is sent to the prediction service.
3. The service runs the ML model homomorphically on the ciphertext. It never sees the plaintext data.
4. The service returns an encrypted prediction result.
5. Only the user can decrypt the result using their key.

## Features

- **Health Risk Prediction**: Predicts health risks based on multiple health metrics
- **Privacy Protection**: Uses FHE to keep sensitive health data encrypted during processing
- **Side-by-Side Comparison**: Shows the difference between traditional and FHE approaches
- **Interactive UI**: User-friendly interface for inputting health data and viewing results
- **Visualization of Encrypted Data**: Shows what encrypted data looks like when sent to the server

## Health Metrics Used

The model uses the following health metrics for prediction:

- Age
- Blood Pressure (Systolic and Diastolic)
- Blood Sugar
- BMI (Body Mass Index)
- Cholesterol Levels (Total, HDL, LDL)
- Smoking Status
- Family History of Heart Disease

## Technology Stack

- **Backend**: Python with Flask
- **Frontend**: HTML, CSS, JavaScript with Bootstrap
- **ML Framework**: scikit-learn for plaintext model
- **FHE Framework**: Concrete-ML (based on Zama's TFHE library)
- **Data**: Synthetic health data generated for demonstration purposes

## Project Structure

```
FHE_Project1/
├── app.py                  # Flask application
├── data/                   # Data directory
│   └── synthetic_health_data.csv  # Synthetic health data
├── models/                 # Model directory
│   ├── plaintext_model.joblib  # Trained plaintext model
│   └── fhe_model/          # FHE model files
├── src/                    # Source code
│   ├── data_generator.py   # Generates synthetic health data
│   ├── train_plaintext_model.py  # Trains the plaintext model
│   ├── compile_fhe_model.py  # Compiles the model for FHE
│   ├── fhe_client.py       # Client-side FHE operations
│   ├── fhe_prediction_service.py  # Server-side FHE operations
│   └── test_inference.py   # Tests both plaintext and FHE inference
├── static/                 # Static files for the web app
│   ├── css/                # CSS files
│   └── js/                 # JavaScript files
├── templates/              # HTML templates
│   ├── base.html           # Base template
│   ├── index.html          # Home page
│   └── results.html        # Results page
├── PLAN.md                 # Detailed implementation steps
├── README.md               # This file
└── .gitignore              # Git ignore file
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/FHE_Project1.git
   cd FHE_Project1
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Generate synthetic data:
   ```
   python src/data_generator.py
   ```

2. Train the plaintext model:
   ```
   python src/train_plaintext_model.py
   ```

3. Compile the FHE model:
   ```
   python src/compile_fhe_model.py
   ```

4. Run the Flask application:
   ```
   python app.py
   ```

5. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Testing

To test the inference with both plaintext and FHE models:
```
python src/test_inference.py
```

## Future Enhancements

- User authentication and account management
- Expanded health metrics and risk factors
- Visualization of health data and risk factors
- Personalized health recommendations
- Mobile application

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Zama](https://www.zama.ai/) for the Concrete-ML library
- [scikit-learn](https://scikit-learn.org/) for the machine learning framework
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the UI components