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
    # --- DATASET SETTINGS ---
    # Change this to your NEW dataset when you are ready
    dataset_name: str = "renemel/compiled-phishing-dataset"
    text_column_candidates: list = ("text", "Email Text", "email_text")

    # --- TRAINING SETTINGS ---
    batch_size: int = 4
    num_epochs: int = 3
    lr: float = 2e-5    # LOWER learning rate to protect old knowledge

    save_dir: str = "models"
    save_every: int = 1
    seed: int = 42

    # --- NEW: RESUME CAPABILITY ---
    # Path to your "smart" model. Set to None to start fresh.
    resume_checkpoint: str = "models/generator_final.pt"
