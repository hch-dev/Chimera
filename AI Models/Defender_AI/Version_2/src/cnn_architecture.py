from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_model(img_height=128, img_width=128):
    model = Sequential([
        # Layer 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Layer 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Layer 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Shuts off 50% of neurons during training to force learning
        Dense(1, activation='sigmoid') # 0=Legit, 1=Phishing
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model