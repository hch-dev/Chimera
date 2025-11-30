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
    dataset_name: str = "RonakAJ/phising_email"
    text_column_candidates: list = ("text", "Email Text", "email_text")

    # --- TRAINING SETTINGS ---
    batch_size: int = 4
    num_epochs: int = 3 # You can keep this low (1-3) for updates
    lr: float = 2e-5    # LOWER LR for updating (so we don't break previous knowledge)

    save_dir: str = "models"
    save_every: int = 1
    seed: int = 42

    # --- NEW: RESUME TRAINING ---
    # Set this to the path of your last model to continue training.
    # Set to None (or empty string) to start from scratch.
    resume_checkpoint: str = "models/generator_final.pt"
