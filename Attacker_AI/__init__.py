# __init__.py
"""
Chimera Attacker AI Engine
───────────────────────────
This package contains:
    - Generator Model (generator_model.py)
    - GAN training scripts (train_gan.py)
    - Trainer logic (trainer.py)
    - Dataset loader (dataset_loader.py)
    - Utility functions (utils.py)
    - Inference interface (inference.py)

The Attacker Engine produces *synthetic phishing-style patterns* ONLY
for training and benchmarking the Defender AI.
"""

from .generator_model import GeneratorModel
from .trainer import AttackerTrainer
from .dataset_loader import AttackerDataset
from .utils import *
