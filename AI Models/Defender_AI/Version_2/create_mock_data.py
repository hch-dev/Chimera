import os
from PIL import Image
import numpy as np
import random

def generate_mock_data(root_dir='training_data', count=50):
    """
    Creates mock image data for Phishing (Class 1) and Legitimate (Class 0).
    """
    phish_dir = os.path.join(root_dir, 'phishing')
    legit_dir = os.path.join(root_dir, 'legitimate')

    os.makedirs(phish_dir, exist_ok=True)
    os.makedirs(legit_dir, exist_ok=True)
    
    print(f"Creating {count} mock images in each of the two categories...")

    for i in range(count):
        # Legitimate: simple green/blue theme (RGB)
        color = (random.randint(50, 150), random.randint(150, 255), random.randint(50, 255))
        img_legit = Image.new('RGB', (random.randint(200, 400), random.randint(300, 600)), color=color)
        img_legit.save(os.path.join(legit_dir, f'legit_{i}.png'))

        # Phishing: simple red/dark theme (RGB)
        color = (random.randint(150, 255), random.randint(0, 100), random.randint(0, 100))
        img_phish = Image.new('RGB', (random.randint(200, 400), random.randint(300, 600)), color=color)
        img_phish.save(os.path.join(phish_dir, f'phish_{i}.png'))
        
    print(f"Mock data creation complete. Total images: {count * 2}")

if __name__ == '__main__':
    generate_mock_data(count=50)