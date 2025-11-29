# configs.py
from dataclasses import dataclass

@dataclass
class GeneratorConfig:
    model_name: str = "gpt2"       # tokenizer model (we'll use tokenizer.vocab_size)
    max_length: int = 64           # sequence length
    d_model: int = 256             # transformer hidden size
    nhead: int = 4                 # transformer attention heads
    num_layers: int = 2            # transformer layers
    dropout: float = 0.1
    device: str = "cpu"            # "cuda" if you have GPU

@dataclass
class TrainingConfig:
    dataset_name: str = "RonakAJ/phising_email"
    text_column_candidates: list = ("text", "Email Text", "email_text")
    batch_size: int = 32
    num_epochs: int = 3
    lr: float = 3e-4
    save_dir: str = "models"
    save_every: int = 1
    seed: int = 42
