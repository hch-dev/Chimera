# inference.py
"""
Inference interface for the Chimera Attacker Generator AI.
Produces *synthetic phishing-like patterns* solely for training
the Defender AI's detection mechanisms.

Provides:
    - load_model()
    - generate()
    - generate_batch()
    - sample_latent()
    - decode output text
"""

import torch
from torch import nn
from typing import List

from configs import ModelConfig, TrainingConfig
from generator_model import GeneratorModel
from utils import (
    log,
    decode_tokens,
    sample_latent,
)


class AttackerInference:
    """
    Loads the trained generator and produces synthetic samples.
    """

    def __init__(self, tokenizer, checkpoint_path=None, device=None):
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load configs
        self.model_cfg = ModelConfig()
        self.training_cfg = TrainingConfig()

        # initialize model
        self.generator = GeneratorModel(
            vocab_size=self.training_cfg.vocab_size,
            embed_dim=self.model_cfg.embed_dim,
            hidden_dim=self.model_cfg.hidden_dim,
            max_length=self.training_cfg.max_length,
        ).to(self.device)

        # optional: load checkpoint
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        self.generator.eval()

    # ---------------------------------------------------------
    #   Load checkpoint
    # ---------------------------------------------------------

    def _load_checkpoint(self, path: str):
        """Load a trained generator checkpoint."""
        try:
            chkpt = torch.load(path, map_location=self.device)
            self.generator.load_state_dict(chkpt["model_state_dict"])
            log(f"Loaded generator checkpoint from {path}")
        except Exception as e:
            log(f"Failed to load checkpoint: {e}")

    # ---------------------------------------------------------
    #   Single text generation
    # ---------------------------------------------------------

    def generate(self, temperature: float = 1.0) -> str:
        """
        Generate a single synthetic sample (string).
        """

        latent = sample_latent(
            batch_size=1,
            latent_dim=self.model_cfg.latent_dim,
            device=self.device,
        )

        with torch.no_grad():
            output_logits = self.generator(latent)  # [1, seq_len, vocab]

        # sample tokens from logits
        token_ids = self._sample_tokens(output_logits, temperature)
        text = decode_tokens(self.tokenizer, token_ids)

        return text.strip()

    # ---------------------------------------------------------
    #   Batch generation
    # ---------------------------------------------------------

    def generate_batch(self, batch_size: int, temperature: float = 1.0) -> List[str]:
        """
        Generate multiple synthetic strings at once.
        """

        latent = sample_latent(
            batch_size=batch_size,
            latent_dim=self.model_cfg.latent_dim,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.generator(latent)   # [B, seq_len, vocab]

        token_ids = self._sample_tokens(logits, temperature)
        decoded = decode_tokens(self.tokenizer, token_ids)

        # normalize return type
        if isinstance(decoded, str):
            return [decoded]
        return [x.strip() for x in decoded]

    # ---------------------------------------------------------
    #   Internal sampling logic
    # ---------------------------------------------------------

    def _sample_tokens(self, logits, temperature: float):
        """
        Apply temperature + softmax sampling to logits and
        return token IDs as Python lists.
        """

        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)

        # sample token per position
        sampled = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(logits.size(0), logits.size(1))

        return sampled.cpu()

    # ---------------------------------------------------------
    #   Convenience: CLI-style usage
    # ---------------------------------------------------------

    def interactive(self, count=5, temperature=0.9):
        """
        Quickly print N samples to terminal.
        """

        texts = self.generate_batch(count, temperature=temperature)
        log("Generated Samples:", verbose=True)
        for i, t in enumerate(texts):
            print(f"[{i}] {t}\n")
