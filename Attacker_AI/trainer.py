# trainer.py
"""
GAN Trainer for the Attacker AI (Generator).
Uses PyTorch for the generator and optionally HuggingFace Transformers
for embedding/tokenization.

This file does NOT modify generator_model.py or train_gan.py.
"""

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from configs import AttackerConfig
from dataset_loader import AttackerDataset
from generator_model import GeneratorModel  # assumed existing
from utils import save_checkpoint, load_checkpoint


class GANTrainer:
    """
    Handles training of the attacker GAN generator.
    Only the generator is implemented; the "discriminator" is conceptual:
        - Defender AI acts as natural adversary
        - Reward/score backprop is allowed through surrogate loss

    The structure is modular so later you can add:
        - Reinforcement learning style adversarial scoring
        - Real discriminator model
        - Defender feedback server
    """

    def __init__(self, config: AttackerConfig):
        self.cfg = config
        self.device = torch.device(self.cfg.generator.device)

        # -----------------------
        # Load tokenizer (HF)
        # -----------------------
        from transformers import AutoTokenizer
        tok_name = self.cfg.training.tokenizer_name or \
                   self.cfg.hf.tokenizer_name or \
                   self.cfg.hf.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if self.cfg.hf.pad_token_as_eos:
            # Needed for GPT-2 style models
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -----------------------
        # Generator Model
        # -----------------------
        self.generator = GeneratorModel(self.cfg.generator)
        self.generator.to(self.device)

        # -----------------------
        # Optimizer
        # -----------------------
        self.optimizer = optim.AdamW(
            self.generator.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay
        )

        # -----------------------
        # Data
        # -----------------------
        self.dataset = AttackerDataset(
            dataset_path=self.cfg.training.dataset_path,
            tokenizer=self.tokenizer,
            max_length=self.cfg.generator.max_length
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True
        )

        # -----------------------
        # Mixed Precision
        # -----------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.training.mixed_precision)

        # -----------------------
        # Loss
        # -----------------------
        self.criterion = nn.CrossEntropyLoss()

    # ============================================
    #  Adversarial Loss Placeholder
    # ============================================
    def adversarial_reward(self, generated_batch):
        """
        Placeholder for adversarial scoring.

        For now:
            - Reward = negative perplexity proxy
        Later:
            - Replace with Defender feedback
            - Or RLHF / PPO reward shaping
        """
        # pseudo scalar reward for now
        reward = torch.rand(generated_batch.size(0), device=self.device)
        return reward

    # ============================================
    #  Training Step
    # ============================================
    def train_step(self, real_input_ids):
        self.generator.train()

        # Noise for generator
        latent = torch.randn(
            real_input_ids.size(0),
            self.cfg.generator.latent_dim,
            device=self.device
        )

        with torch.cuda.amp.autocast(enabled=self.cfg.training.mixed_precision):
            generated_logits = self.generator(latent)

            # Shift for CE loss
            generated_logits = generated_logits[:, :-1, :].contiguous()
            target_ids = real_input_ids[:, 1:].contiguous()

            ce_loss = self.criterion(
                generated_logits.view(-1, generated_logits.size(-1)),
                target_ids.view(-1)
            )

            # Adversarial scoring placeholder
            adv_reward = self.adversarial_reward(real_input_ids)
            adv_loss = -adv_reward.mean()  # maximize reward

            total_loss = ce_loss + adv_loss

        # Backprop
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()

        if self.cfg.training.grad_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.cfg.training.grad_clip
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "adv_loss": adv_loss.item(),
        }

    # ============================================
    #  Training Loop
    # ============================================
    def train(self):
        print("\n=== Starting Attacker GAN Training ===\n")

        for epoch in range(1, self.cfg.training.num_epochs + 1):
            print(f"Epoch {epoch}/{self.cfg.training.num_epochs}")
            epoch_loss = 0

            for batch in self.dataloader:
                batch = batch.to(self.device)
                log = self.train_step(batch)

                epoch_loss += log["loss"]

            print(f"Epoch {epoch} Loss: {epoch_loss:.4f}")

            # Save checkpoint
            if epoch % self.cfg.training.save_every == 0:
                save_path = os.path.join(
                    self.cfg.training.checkpoint_dir,
                    f"generator_epoch_{epoch}.pt"
                )
                save_checkpoint(self.generator, self.optimizer, save_path)
                print(f"Saved checkpoint → {save_path}")

        print("\n=== Training Complete ===\n")

    # ============================================
    #  Sampling
    # ============================================
    @torch.no_grad()
    def sample(self, num_samples=1):
        self.generator.eval()
        latent = torch.randn(num_samples, self.cfg.generator.latent_dim, device=self.device)
        logits = self.generator(latent)
        ids = logits.argmax(dim=-1)

        texts = [self.tokenizer.decode(x, skip_special_tokens=True) for x in ids]
        return texts
