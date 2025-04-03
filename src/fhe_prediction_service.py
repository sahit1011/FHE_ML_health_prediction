import os
from concrete.ml.deployment import FHEModelServer

FHE_MODEL_DIR = "models/fhe_model"

class FHEServer:
    """Simulates the server holding the FHE model for prediction."""
    def __init__(self, model_dir=FHE_MODEL_DIR):
        self.model_dir = model_dir
        self.server = None
        self.load_model()

    def load_model(self):
        """Loads the compiled FHE model from disk."""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"FHE model directory not found: {self.model_dir}")
        print(f"[Server] Loading FHE model from {self.model_dir}...")
        # Initialize the FHEModelServer with the server.zip file
        self.server = FHEModelServer(path_dir=self.model_dir)
        # Load the model
        self.server.load()
        print("[Server] FHE model loaded.")

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

# Example usage (for testing the loading)
if __name__ == "__main__":
    try:
        server = FHEServer()
        print("Server initialized successfully.")
        # We can't run predict here without encrypted data from a client
    except Exception as e:
        print(f"Error initializing server: {e}")