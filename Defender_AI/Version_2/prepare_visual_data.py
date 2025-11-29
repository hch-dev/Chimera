import os
import numpy as np
from PIL import Image
import joblib
from sklearn.model_selection import train_test_split
import random
# Note: The sklearn StandardScaler is used as a stand-in for complex scaling
# If only 0-1 normalization is used, the scaler might be unnecessary, 
# but we save it to fulfill the project requirement for `scaler_v1.pkl`.
from sklearn.preprocessing import StandardScaler 
from create_mock_data import generate_mock_data # Utility to run the mock data generation

# --- Configuration ---
DATA_ROOT = 'training_data'
TARGET_SIZE = (128, 128)  # Required input size for the CNN
SCALER_PATH = 'models/scaler_v1.pkl'
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def load_and_preprocess_images():
    """
    Loads images, resizes them, and prepares the numpy array dataset.
    """
    X_data = [] # Stores image data (features)
    y_labels = [] # Stores labels (0 or 1)
    
    # 1. Load Data from Subdirectories
    categories = {'legitimate': 0, 'phishing': 1}
    
    for folder, label in categories.items():
        folder_path = os.path.join(DATA_ROOT, folder)
        print(f"Processing folder: {folder_path} (Label: {label})")
        
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(folder_path, filename)
                
                try:
                    # Load image and convert to RGB
                    img = Image.open(filepath).convert('RGB')
                    
                    # 2. Resize to Fixed Dimension (128x128)
                    img = img.resize(TARGET_SIZE)
                    
                    # Convert to numpy array
                    img_array = np.array(img)
                    X_data.append(img_array)
                    y_labels.append(label)
                    
                except Exception as e:
                    print(f"Skipping file {filename} due to error: {e}")

    X = np.array(X_data)
    y = np.array(y_labels)

    print(f"\n--- Data Load Complete ---")
    print(f"Total images loaded: {len(X)}")
    print(f"Original shape: {X.shape}") 
    return X, y

def normalize_and_split(X, y):
    """
    Normalizes pixel values (0.0 - 1.0), saves the scaler, and splits the data.
    """
    # 3. Normalize Pixel Values (0-255 -> 0.0-1.0)
    # This is standard simple normalization for images.
    X_normalized = X.astype('float32') / 255.0

    # Reshape the data for simpler standardization if needed,
    # but for CNNs, we often stick to the 0-1 range.
    
    # MOCK: Saving a simple StandardScaler object to fulfill the `scaler_v1.pkl` requirement
    # In reality, a custom ImageScaler might be used, but this demonstrates the serialization concept.
    
    # Reshape for StandardScaler: (Samples, Features)
    X_flat = X_normalized.reshape(X_normalized.shape[0], -1) 
    
    scaler = StandardScaler()
    # We fit the scaler only on the data we will use for TRAINING (80% of data)
    
    # First split for Train (80%) and Temp (20% -> Validation + Test)
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X_normalized, y, test_size=TEST_RATIO, random_state=42, stratify=y
    )

    # Second split for Train (80%) and Validation (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, 
        test_size=(VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)), # 0.1 / 0.9 = 0.111...
        random_state=42, stratify=y_train_temp
    )
    
    # Fit the scaler on the final training data (flat)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    scaler.fit(X_train_flat)
    
    # 5. Save the Scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to: {SCALER_PATH}")


    # Report split sizes
    print("\n--- Data Split Summary ---")
    print(f"Training Set: {len(X_train)} samples ({TRAIN_RATIO*100:.0f}%)")
    print(f"Validation Set: {len(X_val)} samples ({VAL_RATIO*100:.0f}%)")
    print(f"Testing Set: {len(X_test)} samples ({TEST_RATIO*100:.0f}%)")
    
    # The output data X_train, X_val, X_test are ready for the CNN model.
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    # 1. Ensure mock data exists for demonstration
    if not os.path.exists(DATA_ROOT):
        generate_mock_data(DATA_ROOT, count=100)
    
    # 2. Load, Resize, and Convert to Array
    X_raw, y_raw = load_and_preprocess_images()
    
    if len(X_raw) > 0:
        # 3. Normalize, Split, and Save Scaler
        X_train, X_val, X_test, y_train, y_val, y_test = normalize_and_split(X_raw, y_raw)
        
        # Display final shapes
        print("\n--- Final Shapes for CNN Input ---")
        print(f"X_train shape (for model.fit): {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
    else:
        print("No images found to process. Please check the 'training_data' directory.")