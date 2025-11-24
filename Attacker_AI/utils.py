# utils.py
"""
Utility functions for the Attacker AI Engine (Chimera).
Contains:
    - Logging utilities
    - Checkpoint saving/loading
    - Random seed control
    - Sample decoding helpers
    - Safe file/directory handling
"""

import os
import json
import random
import torch
import numpy as np
from datetime import datetime


# ============================================
# 1. Directory + File Utilities
# ============================================

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def timestamp():
    """Return a readable timestamp string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ============================================
# 2. Logging Utilities
# ============================================

def log(message: str, verbose: bool = True):
    """
    Simple log function for terminal + optional logging.
    You can redirect this later into a file if needed.
    """
    if verbose:
        print(f"[LOG {timestamp()}] {message}")


def save_json(obj, path: str):
    """Save any Python dict/list as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    """Load JSON safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found → {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================
# 3. Reproducibility
# ============================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================
# 4. Model Checkpoints
# ============================================

def save_checkpoint(model, optimizer, epoch, path: str):
    """
    Save a PyTorch model checkpoint.
    """
    ensure_dir(os.path.dirname(path))

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)
    log(f"Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path: str, device="cpu"):
    """
    Load a PyTorch checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint does not exist → {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    log(f"Checkpoint loaded → {path}")

    return epoch


# ============================================
# 5. Sampling & Decoding Utilities
# ============================================

def sample_latent(batch_size: int, latent_dim: int, device="cpu"):
    """Generate random noise vectors for the generator."""
    return torch.randn(batch_size, latent_dim, device=device)


def decode_tokens(tokenizer, token_ids):
    """
    Decode a sequence of token IDs into a string.
    Supports batching or single tensors.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    if isinstance(token_ids[0], list):  
        # batch
        return [tokenizer.decode(t, skip_special_tokens=True) for t in token_ids]
    else:
        # single
        return tokenizer.decode(token_ids, skip_special_tokens=True)


# ============================================
# 6. Pretty-print Generated Samples
# ============================================

def preview_samples(strings, count=5):
    """
    Print sample generated strings for quick debugging.
    """
    print("\n=== SAMPLE OUTPUTS =====================")
    for i, s in enumerate(strings[:count]):
        print(f"[{i}] {s}")
    print("=======================================\n")
