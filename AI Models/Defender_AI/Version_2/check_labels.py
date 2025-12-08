from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Check if folder exists first
if not os.path.exists('dataset/train'):
    print("❌ Error: 'dataset/train' folder not found.")
    exit()

# Initialize generator
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    'dataset/train', 
    target_size=(128,128),
    batch_size=1
)

print("\n" + "="*50)
print(f"📊 YOUR MAPPINGS: {generator.class_indices}")
print("="*50)

# Automatic Explanation
phishing_index = generator.class_indices.get('phishing')

if phishing_index == 1:
    print("✅ CORRECT CONFIGURATION:")
    print(" - Phishing is '1'")
    print(" - Legit is '0'")
    print(" -> A HIGH score (close to 1.0) means PHISHING.")
else:
    print("⚠️ SWAPPED CONFIGURATION:")
    print(" - Phishing is '0'")
    print(" - Legit is '1'")
    print(" -> A LOW score (close to 0.0) means PHISHING.")
    print(" -> You must flip your logic in main.py!")