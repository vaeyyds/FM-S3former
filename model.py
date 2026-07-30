import torch
import torch.nn as nn

from .components import EncoderLayer, Encoder
from .film import FiLM
from .patch import PatchEmbedding, PatchOutputHead


class FMS3former(nn.Module):
    def __init__(self, patch_size, chunk_size, d_model, num_heads,
                 num_layers, dim_feedforward, d_block=None,
                 film_hidden=16, use_film=True, film_cond="amp"):
        super().__init__()
        assert chunk_size % patch_size == 0
        assert film_cond in ("amp", "amp_fre")
        self.patch_size = patch_size
        self.chunk_size = chunk_size
        self.num_patches = chunk_size // patch_size
        self.use_film = use_film
        self.film_cond = film_cond
        cond_dim = 1 if film_cond == "amp" else 2
        d_block = d_block or d_model

        self.patch_embed = PatchEmbedding(patch_size, d_model)
        self.film_in = FiLM(d_model, film_hidden, cond_dim) if use_film else None
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    d_model=d_model,
                    n_heads=num_heads,
                    d_block=d_block,
                    d_ff=dim_feedforward,
                    dropout=0.0,
                )
                for _ in range(num_layers)
            ],
        )
        self.film_out = FiLM(d_block, film_hidden, cond_dim) if use_film else None
        self.output_head = PatchOutputHead(d_block, patch_size)

    def _film_cond(self, x, patches):
        B = x.shape[0]
        amp = patches.pow(2).sum(dim=-1).sqrt().mean(dim=2, keepdim=True)
        if self.film_cond == "amp":
            return amp
        xc = torch.complex(x[..., 0], x[..., 1])
        diff = torch.angle(xc[:, 1:] * xc[:, :-1].conj())
        zero = torch.zeros(B, 1, device=x.device, dtype=diff.dtype)
        dphi = torch.cat([zero, diff], dim=1).reshape(
            B, self.num_patches, self.patch_size).mean(dim=2, keepdim=True)
        return torch.cat([amp, dphi], dim=-1)

    def _decode(self, encoded):
        B = encoded.shape[0]
        out = self.output_head(encoded)
        out = out.reshape(B, self.chunk_size, 2)
        return out[:, :, 0], out[:, :, 1]

    def forward(self, x, return_layer_outputs=False, layer_groups=None, aggregate="sum"):
        B = x.shape[0]
        patches = x.reshape(B, self.num_patches, self.patch_size, 2)
        cond = self._film_cond(x, patches) if self.use_film else None
        tokens = self.patch_embed(patches)
        if self.use_film:
            tokens = self.film_in(tokens, cond)
        tokens = tokens + self.pos_embed
        if not return_layer_outputs:
            if aggregate == "backbone":
                encoded = self.encoder(tokens, return_backbone=True)
            else:
                encoded = self.encoder(tokens)
            if self.use_film:
                encoded = self.film_out(encoded, cond)
            return self._decode(encoded)
        encoded, layer_outputs = self.encoder(tokens, return_layer_outputs=True)
        if self.use_film:
            encoded = self.film_out(encoded, cond)
        final_pred = self._decode(encoded)
        if self.use_film:
            layer_outputs = [self.film_out(o, cond) for o in layer_outputs]
        if layer_groups is None:
            layer_groups = [1] * len(layer_outputs)
        assert sum(layer_groups) <= len(layer_outputs)
        group_preds = []
        idx = 0
        for group_size in layer_groups:
            group_sum = sum(layer_outputs[idx: idx + group_size])
            group_preds.append(self._decode(group_sum))
            idx += group_size
        return final_pred, group_preds
