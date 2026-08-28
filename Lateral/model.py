import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    """Simple single-layer linear probe operating on pre-extracted 768-dim features."""
    def __init__(self, in_features=768):
        super().__init__()
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class MLPProbe(nn.Module):
    """Multi-layer perceptron probe with normalization and dropout."""
    def __init__(self, in_features=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)