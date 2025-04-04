from flask import Flask, render_template, request, make_response
from flask_bootstrap import Bootstrap
import numpy as np
import os
import time
import json
from datetime import datetime

# Import our client and server classes
from src.fhe_client import FHEClient
from src.fhe_prediction_service import FHEServer
import joblib

# Configuration
PLAINTEXT_MODEL_FILE = "models/plaintext_model.joblib"
FHE_MODEL_DIR = "models/fhe_model"

# Initialize Flask app
app = Flask(__name__)
Bootstrap(app)

# Load the plaintext model
plaintext_model = None
if os.path.exists(PLAINTEXT_MODEL_FILE):
    plaintext_model = joblib.load(PLAINTEXT_MODEL_FILE)

# Initialize FHE client and server
fhe_client = None
fhe_server = None
try:
    fhe_client = FHEClient(model_dir=FHE_MODEL_DIR)
    fhe_server = FHEServer(model_dir=FHE_MODEL_DIR)
except Exception as e:
    print(f"Error initializing FHE components: {e}")

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    # Get input data from the form
    try:
        # Basic health metrics
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])

        # Blood pressure
        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])

        # Blood tests
        blood_sugar = float(request.form['blood_sugar'])
        total_cholesterol = float(request.form['total_cholesterol'])
        hdl_cholesterol = float(request.form['hdl_cholesterol'])
        ldl_cholesterol = float(request.form['ldl_cholesterol'])

        # Risk factors (checkboxes)
        smoking = 1 if 'smoking' in request.form else 0
        family_history = 1 if 'family_history' in request.form else 0

        # Check if user wants to skip FHE prediction
        skip_fhe = 'skip_fhe' in request.form

        # Create input array with all features
        input_data = np.array([
            [age, systolic_bp, diastolic_bp, blood_sugar, bmi,
             total_cholesterol, hdl_cholesterol, ldl_cholesterol,
             smoking, family_history]
        ], dtype=np.float32)

        # Results dictionary to store all information
        results = {
            'input_data': {
                # Basic health metrics
                'age': age,
                'bmi': bmi,
                # Blood pressure
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                # Blood tests
                'blood_sugar': blood_sugar,
                'total_cholesterol': total_cholesterol,
                'hdl_cholesterol': hdl_cholesterol,
                'ldl_cholesterol': ldl_cholesterol,
                # Risk factors
                'smoking': smoking,
                'family_history': family_history
            },
            'plaintext': {},
            'fhe': {},
            'error': None
        }

        # --- 1. Plaintext Prediction ---
        if plaintext_model is not None:
            start_time_plain = time.time()
            plaintext_prediction = plaintext_model.predict(input_data)
            plaintext_proba = plaintext_model.predict_proba(input_data)
            end_time_plain = time.time()

            results['plaintext'] = {
                'prediction': int(plaintext_prediction[0]),
                'probabilities': plaintext_proba[0].tolist(),
                'inference_time': round(end_time_plain - start_time_plain, 4)
            }
        else:
            results['error'] = "Plaintext model not loaded."

        # --- 2. FHE Prediction ---
        # Initialize FHE results dictionary (empty by default)
        results['fhe'] = {}

        if fhe_client is not None and fhe_server is not None and not skip_fhe:
            # Client encrypts data
            start_time_enc = time.time()
            encrypted_input = fhe_client.encrypt_data(input_data)
            end_time_enc = time.time()
            encryption_time = round(end_time_enc - start_time_enc, 4)

            # Convert the actual encrypted data to a displayable format
            # If it's bytes, convert to hex representation
            if isinstance(encrypted_input, bytes):
                # Get the full hex representation
                encrypted_hex = encrypted_input.hex()
                # Store the first 200 characters for display
                encrypted_sample = encrypted_hex[:200]
                # Store the total length for informational purposes
                encrypted_length = len(encrypted_hex)
            else:
                # If it's not bytes, convert to string
                encrypted_str = str(encrypted_input)
                encrypted_sample = encrypted_str[:200]
                encrypted_length = len(encrypted_str)

            # Get evaluation keys
            evaluation_keys = fhe_client.get_evaluation_keys()

            # Server performs FHE prediction with a timeout
            import threading
            import queue

            # Create a queue for the result
            result_queue = queue.Queue()

            # Define a function to run the prediction in a separate thread
            def run_prediction():
                try:
                    result = fhe_server.predict(encrypted_input, evaluation_keys)
                    result_queue.put((True, result))
                except Exception as e:
                    result_queue.put((False, str(e)))

            # Start the prediction in a separate thread
            start_time_fhe_pred = time.time()
            prediction_thread = threading.Thread(target=run_prediction)
            prediction_thread.daemon = True
            prediction_thread.start()

            # Wait for the prediction to complete with a timeout
            try:
                success, result = result_queue.get(timeout=10)  # 10 second timeout
                if success:
                    encrypted_prediction = result
                else:
                    raise Exception(f"FHE prediction failed: {result}")
            except queue.Empty:
                # If timeout occurs, use plaintext prediction as fallback
                raise Exception("FHE prediction timed out. The computation is taking too long.")
            finally:
                end_time_fhe_pred = time.time()
                inference_time = round(end_time_fhe_pred - start_time_fhe_pred, 4)

            # Client decrypts result
            start_time_dec = time.time()
            decrypted_prediction = fhe_client.decrypt_result(encrypted_prediction)
            end_time_dec = time.time()
            decryption_time = round(end_time_dec - start_time_dec, 4)

            # Get the predicted class from FHE probabilities (index of max value)
            fhe_class = np.argmax(decrypted_prediction[0])

            results['fhe'] = {
                'prediction': int(fhe_class),
                'probabilities': decrypted_prediction[0].tolist(),
                'encryption_time': encryption_time,
                'inference_time': inference_time,
                'decryption_time': decryption_time,
                'total_time': round(encryption_time + inference_time + decryption_time, 4),
                'encrypted_sample': encrypted_sample,
                'encrypted_length': encrypted_length
            }
        else:
            if 'error' not in results:
                results['error'] = "FHE components not loaded."
            else:
                results['error'] += " FHE components not loaded."

        return render_template('results.html', results=results)

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/fhe-privacy')
def fhe_privacy():
    """Render the FHE privacy explanation page."""
    return render_template('fhe_privacy.html')

@app.route('/view_report', methods=['POST'])
def view_report():
    """Generate and download a PDF report of the health assessment."""
    try:
        # Get the results data from the form
        results_data = request.form.get('results_data')
        if not results_data:
            return render_template('error.html', error="No results data provided.")

        # Parse the JSON data
        results = json.loads(results_data)

        # Create a temporary HTML file for the report
        report_html = render_template('report_template.html', results=results, date=datetime.now().strftime("%B %d, %Y"))

        # For now, let's return the HTML directly instead of generating a PDF
        # This will allow us to see the report and debug any issues
        response = make_response(report_html)
        response.headers['Content-Type'] = 'text/html'
        response.headers['Content-Disposition'] = 'inline; filename=health_assessment_report.html'

        # We're returning HTML directly for now
        # In the future, we can implement PDF generation

        return response

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
