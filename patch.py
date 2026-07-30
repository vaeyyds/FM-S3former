import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(patch_size * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        B, N, P, C = x.shape
        return self.mlp(x.reshape(B, N, P * C))


class PatchOutputHead(nn.Module):
    def __init__(self, d_block, patch_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_block, d_block),
            nn.GELU(),
            nn.Linear(d_block, patch_size * 2),
        )

    def forward(self, x):
        return self.mlp(x)
