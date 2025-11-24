# configs.py
"""
Central configuration module for the Attacker (Generator) AI.
Compatible with both PyTorch and HuggingFace Transformers.

You can import any config with:
    from configs import GeneratorConfig, TrainingConfig, HFModelConfig
"""

from dataclasses import dataclass
from typing import Optional


# ============================
# 1. Generator Model Config
# ============================

@dataclass
class GeneratorConfig:
    """
    Configuration for the Generator NN (PyTorch model).
    This is the model that creates adversarial text samples.
    """
    latent_dim: int = 128                  # Input noise dimension
    hidden_dim: int = 512                 # Internal feedforward layers
    num_layers: int = 4                   # Depth of generator network
    dropout: float = 0.1
    vocab_size: int = 30522               # Default BERT tokenizer size
    max_length: int = 64                  # Max length of generated sequence
    activation: str = "gelu"              # Activation fn: gelu, relu, etc.

    device: str = "cuda"                  # cuda / cpu / mps


# ============================
# 2. Training Config
# ============================

@dataclass
class TrainingConfig:
    """
    Settings for the GAN training loop (PyTorch).
    Used by train_gan.py or trainer.py
    """
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    checkpoint_dir: str = "checkpoints/"
    save_every: int = 1                    # Save model every N epochs

    mixed_precision: bool = False          # True → fp16
    grad_clip: float = 1.0                 # Gradient clipping

    # Dataset
    dataset_path: str = "data/training_texts.json"
    tokenizer_name: str = "bert-base-uncased"


# ============================
# 3. HuggingFace Transformer Config
# ============================

@dataclass
class HFModelConfig:
    """
    Configuration specifically for Transformer-based generators
    (BERT, GPT-2, LLaMA, Falcon, etc.)
    """
    model_name: str = "gpt2"              # HF model id
    tokenizer_name: Optional[str] = None  # Optional override
    max_length: int = 64
    pad_token_as_eos: bool = True         # GPT-2 lacks pad token

    # Fine-tuning settings
    lr: float = 5e-5
    warmup_steps: int = 100
    gradient_accumulation: int = 1


# ============================
# 4. Unified Config Getter
# ============================

class AttackerConfig:
    """
    Full combined configuration bundle.
    Example:
        from configs import AttackerConfig
        cfg = AttackerConfig()
        print(cfg.generator.hidden_dim)
    """
    def __init__(self):
        self.generator = GeneratorConfig()
        self.training = TrainingConfig()
        self.hf = HFModelConfig()

    def __repr__(self):
        return (
            "AttackerConfig(\n"
            f"  Generator = {self.generator},\n"
            f"  Training  = {self.training},\n"
            f"  HFModel   = {self.hf}\n"
            ")"
        )
