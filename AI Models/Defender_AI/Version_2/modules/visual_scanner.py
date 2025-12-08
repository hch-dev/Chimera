import os
import numpy as np
import tensorflow as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# Settings matching your training
IMG_SIZE = 128

class VisualScanner:
    def __init__(self):
        # 1. Load the trained model
        # We look for the model in the 'models' folder relative to the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'visual_model.h5')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}. Run train_model.py first!")

        print(f"Loading Model: {model_path}")
        self.model = load_model(model_path)

    def capture_screenshot(self, url, save_path="temp_scan.png"):
        """Captures a headless screenshot of the website."""
        options = Options()
        options.add_argument("--headless") # Run in background
        options.add_argument("--window-size=1280,1024")
        options.add_argument("--ignore-certificate-errors")

        # Initialize WebDriver (Chrome)
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            time.sleep(2) # Wait for render
            driver.save_screenshot(save_path)
            driver.quit()
            return True
        except Exception as e:
            print(f"Screenshot Error: {e}")
            driver.quit()
            return False

    def get_visual_score(self, url):
        """
        Returns a phishing score from 0 to 100.
        Higher = More likely Phishing.
        """
        temp_img = "temp_scan.png"

        # 1. Capture Image
        if not self.capture_screenshot(url, temp_img):
            return 0 # Failed to capture, return safe (or handle as error)

        try:
            # 2. Preprocess for CNN (Resize -> Array -> Normalize)
            img = load_img(temp_img, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Make batch of 1
            img_array = img_array / 255.0 # Normalize pixel values

            # 3. Predict
            prediction = self.model.predict(img_array)[0][0]

            # Clean up
            if os.path.exists(temp_img):
                os.remove(temp_img)

            # 4. Return Score (0-100)
            # Assuming '1' is Phishing (from your check_labels.py logic)
            return float(prediction * 100)

        except Exception as e:
            print(f"Prediction Error: {e}")
            return 0
