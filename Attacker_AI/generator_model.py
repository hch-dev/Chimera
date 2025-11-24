class FeatureGenerator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, max_perturb: float = 0.15):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_perturb = max_perturb

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raw_perturb = self.net(x)
        perturb = torch.clamp(raw_perturb, -self.max_perturb, self.max_perturb)
        adv = x + perturb
        adv = torch.clamp(adv, 0.0, 1.0)

        metadata = {
            "synthetic": True,
            "perturb_norm": torch.norm(perturb, p=2).item()
        }
        return adv, metadata

def build_generator(config: Dict[str, Any]) -> FeatureGenerator:
    return FeatureGenerator(
        feature_dim=config.get("feature_dim", 32),
        hidden_dim=config.get("hidden_dim", 128),
        max_perturb=config.get("max_perturb", 0.15)
    )
