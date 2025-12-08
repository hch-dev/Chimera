import os 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.cnn_architecture import create_model
from src.utils import logger

IMG_SIZE = 128
BATCH_SIZE = 32

def train():
    logger.info("Initializing Training...")

    # Data Augmentation (Makes the model smarter by rotating/zooming images slightly)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.15
    )
    
    train_gen = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    val_gen = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    model = create_model(IMG_SIZE, IMG_SIZE)
    
    logger.info("Starting Epochs...")
    model.fit(train_gen, validation_data=val_gen, epochs=15)
    
    # Save Model
    if not os.path.exists('models'): os.makedirs('models')
    model.save('models/visual_model.h5')
    logger.info("Model Saved to models/visual_model.h5")

if __name__ == "__main__":
    train()