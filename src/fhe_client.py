import numpy as np
import os
from concrete.ml.deployment import FHEModelClient

# We need to load the client parameters saved during compilation
# These are typically stored alongside the model circuit
FHE_MODEL_DIR = "models/fhe_model"

class FHEClient:
    """Simulates the client encrypting data and decrypting results."""
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
        print("[Client] Step 1: Quantizing data (converting float to integer)")
        # The quantization step is handled internally by the client.quantize_encrypt_serialize method
        # But we can explain what's happening
        print("[Client] Quantization maps floating-point values to integers using a scale factor")
        print("[Client] For example, if scale=100, then 0.5 becomes 50, 1.2 becomes 120, etc.")

        print("[Client] Step 2: Encrypting quantized data")
        print("[Client] Each value is encrypted using the TFHE scheme with the client's secret key")
        print("[Client] The encryption is probabilistic, so the same value encrypts differently each time")

        print("[Client] Step 3: Serializing encrypted data for transmission")
        # Use the FHEModelClient to quantize, encrypt, and serialize the data
        encrypted_data = self.client.quantize_encrypt_serialize(data)

        # Print some information about the encrypted data
        if isinstance(encrypted_data, bytes):
            print(f"[Client] Encrypted data size: {len(encrypted_data)} bytes")
            print(f"[Client] Original data size: {data.nbytes} bytes")
            print(f"[Client] Encryption expansion factor: {len(encrypted_data) / data.nbytes:.2f}x")

        print("[Client] Data encrypted and serialized.")
        return encrypted_data

    def decrypt_result(self, encrypted_result):
        """Decrypts the prediction received from the server."""
        print("[Client] Received encrypted result. Decrypting...")

        print("[Client] Step 1: Deserializing encrypted result")
        print("[Client] Converting the bytes back to an encrypted tensor format")

        print("[Client] Step 2: Decrypting result using secret key")
        print("[Client] The secret key is used to remove the encryption layer")

        print("[Client] Step 3: Dequantizing result (converting integer back to float)")
        print("[Client] The integer values are converted back to floating-point using the scale factor")

        # Use the FHEModelClient to deserialize, decrypt, and dequantize the result
        decrypted_result = self.client.deserialize_decrypt_dequantize(encrypted_result)

        print(f"[Client] Result decrypted: {decrypted_result}")
        print(f"[Client] Probabilities: Low Risk = {decrypted_result[0][0]:.4f}, High Risk = {decrypted_result[0][1]:.4f}")
        print(f"[Client] Predicted class: {'Low Risk (0)' if decrypted_result[0][0] > decrypted_result[0][1] else 'High Risk (1)'}")

        return decrypted_result

    def get_evaluation_keys(self):
        """Returns the serialized evaluation keys to be sent to the server."""
        return self.serialized_evaluation_keys

# Example usage (for testing)
if __name__ == "__main__":
    try:
        client = FHEClient()
        print("Client initialized successfully.")

        # Example data point (needs 4 features: Age, Systolic_BP, Diastolic_BP, Blood_Sugar)
        # IMPORTANT: Ensure the data type is float32 for consistency
        sample_data = np.array([[55.0, 145.0, 95.0, 130.0]], dtype=np.float32)

        encrypted_input = client.encrypt_data(sample_data)
        print(f"\nEncrypted input (type): {type(encrypted_input)}")

        # In a real scenario, send `encrypted_input` to the server
        # and receive `encrypted_output` back.
        # Here we simulate a dummy encrypted output for decryption testing.
        # (We can't directly get this without the server running prediction)

        # Dummy encrypted value (won't decrypt correctly, just for flow test)
        # In a real test, we'd need the server's output.
        # For now, let's just test decryption with something arbitrary (if possible)
        # Usually decrypt needs the exact encrypted structure.

        # Let's try encrypting a dummy prediction (e.g., 0 or 1)
        # This is NOT the real workflow, just to test decrypt syntax
        dummy_pred_clear = np.array([[1]], dtype=np.uint64) # FHE often returns integers
        encrypted_dummy_output = client.client_params.encrypt(dummy_pred_clear, client.keys)

        if encrypted_dummy_output is not None:
             decrypted_output = client.decrypt_result(encrypted_dummy_output)
             print(f"\nDecrypted dummy output: {decrypted_output}")
        else:
             print("\nCould not create dummy encrypted output for decryption test.")

    except Exception as e:
        print(f"Error initializing or testing client: {e}")