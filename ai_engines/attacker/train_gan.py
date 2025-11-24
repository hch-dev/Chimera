"""
train_gan.py
-------------
Trains the Generator ("attacker") and Discriminator ("defender")
in an adversarial training loop over sanitized feature vectors.

GOAL:
- Strengthen the Defender against synthetic adversarial feature variations.
- Generator produces safe, abstract, non-actionable adversarial vectors.
- Discriminator learns robustness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from generator_model import build_generator


class DummyDefender(nn.Module):
    """
    Placeholder for the real Defender model.
    In production, this gets replaced with the actual trained classifier.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def load_training_data() -> torch.utils.data.DataLoader:
    """
    Returns a dataloader of safe feature vectors.
    These are sanitized numerical vectors only.
    """
    # Placeholder dataset — replace with real sanitized feature vectors
    dummy_data = torch.rand(512, 32)         # 512 samples, 32 features
    labels = torch.randint(0, 2, (512,))     # binary labels: 0 / 1

    dataset = torch.utils.data.TensorDataset(dummy_data, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


def train_attacker(config: Dict[str, Any]) -> None:
    feature_dim = config.get("feature_dim", 32)
    epochs = config.get("epochs", 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    generator = build_generator(config).to(device)
    defender = DummyDefender(feature_dim).to(device)

    # Optimizers
    g_opt = optim.Adam(generator.parameters(), lr=1e-4)
    d_opt = optim.Adam(defender.parameters(), lr=1e-4)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    dataloader = load_training_data()

    # Training loop
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # ---------------------
            # 1. Train Defender
            # ---------------------
            d_opt.zero_grad()

            adv_x, _ = generator(batch_x)
            logits = defender(adv_x.detach())  # detach so D doesn't train G

            loss_d = criterion(logits, batch_y)
            loss_d.backward()
            d_opt.step()

            # ---------------------
            # 2. Train Generator
            # ---------------------
            g_opt.zero_grad()

            adv_x, _ = generator(batch_x)
            logits_adv = defender(adv_x)

            # Binary-only label flip: 0 ↔ 1
            flipped_labels = 1 - batch_y

            loss_g = criterion(logits_adv, flipped_labels)
            loss_g.backward()
            g_opt.step()

        print(
            f"Epoch {epoch + 1} — "
            f"D_loss: {loss_d.item():.4f} | "
            f"G_loss: {loss_g.item():.4f}"
        )

    print("Adversarial training completed (synthetic-only).")


if __name__ == "__main__":
    config = {
        "feature_dim": 32,
        "hidden_dim": 128,
        "epochs": 5,
    }

    train_attacker(config)
