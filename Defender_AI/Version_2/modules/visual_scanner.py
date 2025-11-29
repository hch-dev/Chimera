import os
import numpy as np
from PIL import Image
from log import LOG
# Import relative path utility
from utils.browser_ops import capture_screenshot, cleanup_screenshot, SCREENSHOT_PATH

# MOCK PLACEHOLDERS for model and scaler files
MODEL_PATH = "models/visual_net_v1.h5" 
SCALER_PATH = "models/scaler_v1.pkl"

class VisualScanner:
    """
    Performs dynamic visual analysis using a Convolutional Neural Network (CNN) (New V2 Logic).
    MOCK: The model loading and prediction are simulated.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False # Initialize flag
        LOG.info("VisualScanner (V2) initialized.")
        self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        """
        MOCK: Simulates loading the trained CNN model and a feature scaler.
        """
        if not os.path.exists(MODEL_PATH):
             # Ensure the 'models' directory exists before creating files
             os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
             
             LOG.warning(f"MOCK: Model file not found at {MODEL_PATH}. Creating placeholder files.")
             try:
                 with open(MODEL_PATH, 'w') as f:
                     f.write("CNN Model Placeholder")
                 with open(SCALER_PATH, 'w') as f:
                     f.write("Scaler Placeholder")
             except Exception as e:
                 LOG.error(f"Failed to create mock files: {e}")
                 return # Exit if file creation fails

        # MOCK: Set a flag to indicate 'model loaded' successfully
        self.model_loaded = True
        LOG.info("V2 Model and Scaler MOCK: Successfully initialized placeholder models.")


    def _preprocess_image(self, image_path):
        """
        Preprocesses the screenshot image for CNN input.
        """
        if not os.path.exists(image_path):
            LOG.error(f"Image not found at {image_path}")
            return None

        try:
            img = Image.open(image_path).convert('RGB')
            # Resize is critical for CNN input consistency
            img = img.resize((128, 128)) 
            # Normalize pixel values
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Add batch dimension: (1, 128, 128, 3)
            return np.expand_dims(img_array, axis=0)

        except Exception as e:
            LOG.error(f"Error preprocessing image: {e}")
            return None

    def get_visual_score(self, url):
        """
        Captures the screenshot and runs the CNN prediction (MOCK).
        Returns a score from 0 (Safe) to 100 (High Risk).
        """
        if not self.model_loaded:
            LOG.error("V2 Model not loaded. Returning fallback score.")
            return 50 

        image_path = None
        try:
            # Step 1: Capture the visual state of the page
            image_path = capture_screenshot(url)
            if not image_path:
                return 50 # Neutral fallback if capture fails

            # Step 2: Preprocess the image
            processed_image = self._preprocess_image(image_path)
            if processed_image is None:
                return 50 # Neutral fallback if preprocessing fails

            # Step 3: Run the CNN model prediction (MOCK)
            LOG.info("MOCK: Running simulated CNN prediction...")
            
            # --- MOCK PREDICTION START ---
            # Simulate a result based on keywords in the URL
            lower_url = url.lower()
            if 'phish' in lower_url or 'secure.login.verify' in lower_url:
                # High risk simulation
                risk_score = np.random.randint(90, 100)
                LOG.warning(f"MOCK: Visual analysis detected high visual risk ({risk_score})")
            elif 'google' in lower_url or 'microsoft' in lower_url or 'wikipedia' in lower_url:
                # Low risk simulation for known legitimate sites
                risk_score = np.random.randint(5, 40)
            else:
                # Medium/uncertain risk
                risk_score = np.random.randint(40, 70) 

            # Ensure score is 0-100 and cast to int
            visual_risk_score = int(np.clip(risk_score, 0, 100))
            # --- MOCK PREDICTION END ---

            LOG.info(f"V2 Visual CNN Score for '{url}': {visual_risk_score}/100")
            return visual_risk_score

        except Exception as e:
            LOG.error(f"Critical error during V2 Visual scanning: {e}")
            return 50 # Neutral fallback score

        finally:
            if image_path:
                cleanup_screenshot(image_path)