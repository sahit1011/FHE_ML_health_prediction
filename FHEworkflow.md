# Fully Homomorphic Encryption (FHE) Workflow

This document provides a detailed explanation of the Fully Homomorphic Encryption (FHE) workflow implemented in this project, from encryption on the client side to inference on the server and decryption back on the client.

## Table of Contents

1. [Overview of the FHE Workflow](#overview-of-the-fhe-workflow)
2. [Client-side Encryption](#1-client-side-encryption)
3. [Secure Transmission](#2-secure-transmission)
4. [Homomorphic Inference on the Server](#3-homomorphic-inference-on-the-server)
5. [Client-side Decryption](#4-client-side-decryption)
6. [Complete End-to-End Workflow](#complete-end-to-end-workflow)
7. [Technical Details of the FHE Implementation](#technical-details-of-the-fhe-implementation)
8. [Security Guarantees](#security-guarantees)
9. [Performance Considerations](#performance-considerations)
10. [Summary](#summary-of-the-fhe-workflow)

## Overview of the FHE Workflow

The FHE workflow in this project follows these main steps:

1. **Client-side Encryption**: The client encrypts health data using their private key
2. **Secure Transmission**: Encrypted data is sent to the server
3. **Homomorphic Inference**: The server performs computations on the encrypted data
4. **Return Encrypted Results**: The server returns encrypted prediction results
5. **Client-side Decryption**: The client decrypts the results using their private key

![FHE Workflow Diagram](static/images/fhe_workflow.png)

## 1. Client-side Encryption

### Key Generation

The process begins in the `FHEClient` class in `src/fhe_client.py`:

```python
def __init__(self, model_dir=FHE_MODEL_DIR):
    self.model_dir = model_dir
    # Create a temporary directory for keys if needed
    self.key_dir = os.path.join(self.model_dir, "keys")
    os.makedirs(self.key_dir, exist_ok=True)
    
    print("[Client] Loading FHE client configuration...")
    # Initialize the FHEModelClient with the client.zip file
    self.client = FHEModelClient(path_dir=self.model_dir, key_dir=self.key_dir)
    print("[Client] FHE client configuration loaded.")
    
    # Get the serialized evaluation keys that will be sent to the server
    self.serialized_evaluation_keys = self.client.get_serialized_evaluation_keys()
    print("[Client] FHE keys generated.")
```

What happens here:
1. The client loads the FHE configuration from `client.zip` (created during model compilation)
2. It initializes an `FHEModelClient` from Concrete-ML's deployment module
3. The client generates cryptographic keys:
   - **Secret key**: Stays with the client for encryption/decryption
   - **Evaluation keys**: Sent to the server to enable homomorphic operations

### Data Encryption

When the client wants to make a prediction, it encrypts the health data:

```python
def encrypt_data(self, data):
    """Encrypts the input data vector."""
    # Ensure data is a numpy array of the correct type (float32)
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Data should be shaped correctly (e.g., a single sample as a 2D array)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    print(f"[Client] Encrypting data: {data}")
    # Use the FHEModelClient to quantize, encrypt, and serialize the data
    encrypted_data = self.client.quantize_encrypt_serialize(data)
    print("[Client] Data encrypted and serialized.")
    return encrypted_data
```

What happens here:
1. The input data is converted to the correct format (numpy array of type float32)
2. The `quantize_encrypt_serialize` method performs three important steps:
   - **Quantization**: Converts floating-point values to integers (required for FHE)
   - **Encryption**: Encrypts the quantized data using the client's secret key
   - **Serialization**: Converts the encrypted data to bytes for transmission

## 2. Secure Transmission

In a real-world scenario, the encrypted data would be sent over a network. In our Flask application, this happens in the `predict` route in `app.py`:

```python
# Client encrypts data
start_time_enc = time.time()
encrypted_input = fhe_client.encrypt_data(input_data)
end_time_enc = time.time()
encryption_time = round(end_time_enc - start_time_enc, 4)

# Get evaluation keys
evaluation_keys = fhe_client.get_evaluation_keys()

# Server performs FHE prediction
start_time_fhe_pred = time.time()
encrypted_prediction = fhe_server.predict(encrypted_input, evaluation_keys)
end_time_fhe_pred = time.time()
inference_time = round(end_time_fhe_pred - start_time_fhe_pred, 4)
```

What happens here:
1. The client encrypts the health data
2. The client provides evaluation keys to the server
3. Both the encrypted data and evaluation keys are sent to the server
4. The server performs the prediction and returns encrypted results

## 3. Homomorphic Inference on the Server

The server-side processing happens in the `FHEServer` class in `src/fhe_prediction_service.py`:

```python
def predict(self, encrypted_input, evaluation_keys):
    """Performs FHE prediction on encrypted data."""
    if self.server is None:
        raise RuntimeError("FHE model is not loaded.")

    print("[Server] Received encrypted input. Running FHE prediction...")
    # Use the FHEModelServer to run the prediction on the encrypted data
    # We need both the encrypted input and the evaluation keys from the client
    encrypted_prediction = self.server.run(encrypted_input, evaluation_keys)

    print("[Server] FHE prediction complete. Returning encrypted result.")
    return encrypted_prediction
```

What happens here:
1. The server receives the encrypted data and evaluation keys
2. It uses the `FHEModelServer.run()` method to perform homomorphic operations
3. Under the hood, this executes the compiled FHE circuit on the encrypted data
4. The server never decrypts the data or sees the actual values
5. The result is an encrypted prediction that only the client can decrypt

### How the FHE Model Works

The FHE model was created during the compilation process in `src/compile_fhe_model.py`:

```python
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
```

What happens during compilation:
1. A logistic regression model is trained on plaintext data
2. The model is compiled into an FHE-compatible circuit
3. During compilation, the model is:
   - **Quantized**: Floating-point operations are converted to integer operations
   - **Transformed**: Complex operations are replaced with FHE-friendly alternatives
   - **Optimized**: The circuit is optimized for FHE performance
4. The compiled model is saved as two components:
   - `client.zip`: Contains parameters for encryption/decryption
   - `server.zip`: Contains the FHE circuit for homomorphic inference

## 4. Client-side Decryption

After the server returns the encrypted prediction, the client decrypts it:

```python
def decrypt_result(self, encrypted_result):
    """Decrypts the prediction received from the server."""
    print("[Client] Received encrypted result. Decrypting...")
    # Use the FHEModelClient to deserialize, decrypt, and dequantize the result
    decrypted_result = self.client.deserialize_decrypt_dequantize(encrypted_result)
    print(f"[Client] Result decrypted: {decrypted_result}")
    return decrypted_result
```

What happens here:
1. The client receives the encrypted prediction from the server
2. The `deserialize_decrypt_dequantize` method performs three steps:
   - **Deserialization**: Converts the bytes back to an encrypted tensor
   - **Decryption**: Uses the client's secret key to decrypt the result
   - **Dequantization**: Converts the integer values back to floating-point

## Complete End-to-End Workflow

The complete workflow can be traced through the `test_inference.py` file:

```python
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
```

## Technical Details of the FHE Implementation

### Encryption Scheme

This project uses the TFHE (Torus Fully Homomorphic Encryption) scheme through Concrete-ML, which is built on top of Zama's TFHE library. TFHE is particularly well-suited for machine learning because:

1. It supports both arithmetic and boolean operations
2. It has relatively fast bootstrapping (refreshing ciphertexts)
3. It allows for precise control over the noise growth

### Quantization

FHE operates on integers, not floating-point numbers. Concrete-ML handles the quantization process:

1. During compilation, it determines the optimal quantization parameters
2. Before encryption, floating-point inputs are converted to integers
3. After decryption, integer results are converted back to floating-point

### Circuit Compilation

The compilation process converts the logistic regression model into a circuit of FHE-compatible operations:

1. The weights and biases are quantized
2. The linear combination (dot product) is implemented using homomorphic addition and multiplication
3. The sigmoid activation function is approximated using polynomials or lookup tables
4. The circuit is optimized to minimize the number of operations and noise growth

## Security Guarantees

The security of the FHE implementation comes from:

1. **Secret Key**: Only the client has the secret key needed for decryption
2. **Evaluation Keys**: Allow the server to perform computations without decryption
3. **Noise Management**: The FHE scheme carefully manages noise to ensure correct results
4. **Semantic Security**: The encryption is probabilistic, so the same input encrypts to different ciphertexts

## Performance Considerations

FHE operations are computationally intensive:

1. **Encryption**: Relatively fast (milliseconds)
2. **Homomorphic Inference**: The most time-consuming part (seconds to minutes)
3. **Decryption**: Relatively fast (milliseconds)

Our test results show that the entire process takes only a fraction of a second for a simple logistic regression model, which is quite efficient for FHE.

## Summary of the FHE Workflow

1. **Model Compilation (One-time Setup)**:
   - Train a logistic regression model on plaintext data
   - Compile the model into an FHE-compatible circuit
   - Save client and server components

2. **Client-side Encryption**:
   - Generate cryptographic keys
   - Quantize and encrypt health data
   - Send encrypted data and evaluation keys to the server

3. **Server-side Homomorphic Inference**:
   - Receive encrypted data and evaluation keys
   - Run the FHE circuit on the encrypted data
   - Return encrypted prediction results

4. **Client-side Decryption**:
   - Receive encrypted prediction results
   - Decrypt and dequantize the results
   - Interpret the prediction (e.g., low risk vs. high risk)

This workflow ensures that sensitive health data remains encrypted throughout the entire prediction process, providing strong privacy guarantees while still enabling useful machine learning predictions.

## References

- [Concrete-ML Documentation](https://docs.zama.ai/concrete-ml/)
- [TFHE: Fast Fully Homomorphic Encryption over the Torus](https://eprint.iacr.org/2018/421.pdf)
- [Zama's TFHE-rs Library](https://github.com/zama-ai/tfhe-rs)
