# configs.py
from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    model_name: str = "gpt2"
    max_length: int = 128          # Emails are longer than URLs
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    device: str = "cpu"

@dataclass
class TrainingConfig:
    # --- DATASET SETTINGS ---
    dataset_name: str = "renemel/compiled-phishing-dataset"

    # 'text' is the correct body column for renemel
    text_column_candidates: tuple = ("text", "body", "email", "content")

    # --- TRAINING SETTINGS ---
    batch_size: int = 4   # Keep low for CPU safety
    num_epochs: int = 3   # 3 Epochs is standard for fine-tuning
    lr: float = 2e-5      # Low Learning Rate

    # Save to a specific folder for Email models
    save_dir: str = "models_email"
    save_every: int = 1
    seed: int = 42

    # --- RESUME CAPABILITY ---
    resume_checkpoint: str = "models_email/generator_final.pt"
