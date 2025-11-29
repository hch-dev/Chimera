import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import joblib

# --- Configuration ---
TARGET_SIZE = (128, 128)
INPUT_SHAPE = TARGET_SIZE + (3,) # (128, 128, 3) for RGB images
MODEL_PATH = 'models/visual_net_v1.h5'
SCALER_PATH = 'models/scaler_v1.pkl'
DATA_ROOT = 'training_data' # Assumes this folder exists from Part 1

# --- Utility to Load Preprocessed Data (Simplified) ---

def load_preprocessed_data():
    """
    Loads the image data previously saved or, for simplicity, reloads and splits 
    the data based on the structure created in the previous step.
    
    NOTE: In a real environment, you would save the numpy arrays (X_train, etc.) 
    to disk in Part 1 and load them here. This function re-runs the splitting for demo.
    """
    print("Reloading and splitting data for training...")
    
    X_data = [] 
    y_labels = [] 
    categories = {'legitimate': 0, 'phishing': 1}
    
    # Reload all images 
    for folder, label in categories.items():
        folder_path = os.path.join(DATA_ROOT, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(folder_path, filename)
                try:
                    img = Image.open(filepath).convert('RGB').resize(TARGET_SIZE)
                    img_array = np.array(img).astype('float32') / 255.0 # Normalize 0-1
                    X_data.append(img_array)
                    y_labels.append(label)
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
                    
    X = np.array(X_data)
    y = np.array(y_labels)

    if len(X) == 0:
        raise ValueError("No images loaded. Ensure 'training_data' directory exists and has content.")

    # Re-split (using the same random state as Part 1 for consistency)
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, 
        test_size=(0.1 / 0.9), # 10% of total data
        random_state=42, stratify=y_train_temp
    )
    
    print(f"Data ready. Training samples: {len(X_train)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# --- Step 4: Define the CNN Architecture ---

def build_cnn_model(input_shape):
    """
    Defines a simple Convolutional Neural Network model for image classification.
    """
    model = keras.Sequential([
        # 1. Convolutional Block 1
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),

        # 2. Convolutional Block 2
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # 3. Dense (Classification) Block
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5), # Regularization to prevent overfitting
        keras.layers.Dense(128, activation='relu'),
        
        # Output Layer (Binary Classification: Phishing or Legitimate)
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model


# --- Step 5 & 6: Train and Save ---

def train_and_save_model(X_train, X_val, y_train, y_val):
    """
    Compiles, trains, and saves the final model artifact.
    """
    model = build_cnn_model(INPUT_SHAPE)
    
    # Compile the model
    # Using 'adam' optimizer and 'binary_crossentropy' for binary (0 or 1) classification
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    print("\n--- Starting Model Training (Mock Training Cycle) ---")
    
    # Train the model (Using a small number of epochs for quick demonstration)
    history = model.fit(
        X_train, y_train,
        epochs=5, # Typically requires 20-50 epochs
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    print("\n--- Training Complete ---")
    
    # Evaluate on the test set (optional but recommended)
    # loss, acc = model.evaluate(X_test, y_test, verbose=0)
    # print(f"Test Accuracy: {acc*100:.2f}%")

    # Step 6: Save the Model Artifact (REAL .h5 file)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nSuccessfully saved the trained CNN model to: {MODEL_PATH}")

    # MOCK: Clean up the mock data folder after training (optional)
    # os.remove(SCALER_PATH) # If you want to clean up scaler before final placement

if __name__ == '__main__':
    # 1. Load and Split Data
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    except ValueError as e:
        print(f"FATAL ERROR: {e}")
        print("Please ensure you run 'prepare_visual_data.py' first to create the training data.")
        exit()

    # 2. Train and Save
    train_and_save_model(X_train, X_val, y_train, y_val)
    
    print("\n*******************************************************************")
    print("STEP 6 COMPLETE: The real 'models/visual_net_v1.h5' file is now saved.")
    print("*******************************************************************")