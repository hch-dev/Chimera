# configs.py
from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    # Changed from GeneratorURLConfig to match generator_inference.py
    model_name: str = "gpt2"
    max_length: int = 64
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    device: str = "cpu"  # Change to "cuda" if you have a GPU

@dataclass
class TrainingConfig:
    # Changed from TrainingURLConfig to match train_gan.py
    dataset_name: str = "Mitake/PhishingURLsANDBenignURLs"
    text_column_candidates: tuple = ("url", "URL", "Url")
    label_column_candidates: tuple = ("label", "Label")
    batch_size: int = 32
    num_epochs: int = 3
    lr: float = 3e-4
    save_dir: str = "models_url"
    save_every: int = 1
    seed: int = 42
