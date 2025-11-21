# train_cnn.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import joblib
# ==================================================
# CONFIG
# ==================================================
MAX_WORDS = 10000  # vocabulary size
MAX_LEN = 300  # max length for URL encoding
MODEL_PATH = "models/cnn_model.h5"
TOKENIZER_PATH = "models/cnn_tokenizer.pkl"
# ==================================================
# 1. LOAD DATA
# ==================================================
def load_dataset():
    """
    Replace this implementation with your real dataset loading logic.
    Should return: list_of_urls, list_of_labels
    where labels are 0/1 for benign/phishing.
    """
    # --- Dummy Example (replace with CSV, database loading etc.) ---
    urls = [
        "http://secure-paypal-login.com/auth/update",
        "https://www.google.com/search?q=openai",
        "http://verify-apple-account-security.com/login"
    ]
    labels = [1, 0, 1]  # 1=phishing, 0=safe
    return urls, labels
# ==================================================
# 2. PREPROCESS
# ==================================================
def preprocess_data(urls):
    """
    Tokenize URLs → integer sequences → padded vectors.
    Saves tokenizer for inference phase.
    """
    tokenizer = Tokenizer(num_words=MAX_WORDS, char_level=True)
    tokenizer.fit_on_texts(urls)
    sequences = tokenizer.texts_to_sequences(urls)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    # Save tokenizer
    joblib.dump(tokenizer, TOKENIZER_PATH)
    return padded
# ==================================================
# 3. BUILD MODEL
# ==================================================
def build_cnn_model():
    model = Sequential([
        Conv1D(128, 5, activation='relu', input_shape=(MAX_LEN,)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # probability output
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
# ==================================================
# 4. TRAIN MODEL
# ==================================================
def train():
    # Load dataset
    urls, labels = load_dataset()
    # Preprocess
    X = preprocess_data(urls)
    y = np.array(labels)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Build model
    model = build_cnn_model()
    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
    # Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=10,
        callbacks=[early_stop],
        verbose=1
    )
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nCNN Test Accuracy: {acc:.4f}")
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Tokenizer saved to {TOKENIZER_PATH}")
# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    train()
