# configs.py
from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    model_name: str = "gpt2"
    max_length: int = 64           # URLs are short
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    device: str = "cpu"

@dataclass
class TrainingConfig:
    # --- DATASET SETTINGS ---
    dataset_name: str = "cybersectony/PhishingEmailDetectionv2.0"

    # We will extract URLs from the email body column ('content' or 'text')
    text_column_candidates: tuple = ("content", "text", "body", "email")

    # --- TRAINING SETTINGS ---
    batch_size: int = 4   # Keep low for CPU safety
    num_epochs: int = 3   # 3 Epochs
    lr: float = 2e-5      # Low Learning Rate

    save_dir: str = "models_url"
    save_every: int = 1
    seed: int = 42

    # --- RESUME CAPABILITY ---
    resume_checkpoint: str = "models_url/generator_final.pt"
