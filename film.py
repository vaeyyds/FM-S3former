import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, d_model: int, hidden: int = 16, cond_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta
