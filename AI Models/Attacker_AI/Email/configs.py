# configs.py
from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    model_name: str = "gpt2"
    max_length: int = 64
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    device: str = "cpu"

@dataclass
class TrainingConfig:
    dataset_name: str = "Mitake/PhishingURLsANDBenignURLs"
    text_column_candidates: tuple = ("url", "URL", "Url")

    # --- CRITICAL SETTINGS ---
    batch_size: int = 4  # Small batch size to prevent RAM crashes
    num_epochs: int = 3
    lr: float = 2e-5     # Low learning rate for fine-tuning

    save_dir: str = "models_url"
    save_every: int = 1
    seed: int = 42

    # --- RESUME CAPABILITY ---
    resume_checkpoint: str = "models_url/generator_final.pt"
