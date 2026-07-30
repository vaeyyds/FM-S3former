import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, attention_dropout=0.1):
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = FullAttention(attention_dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values)
        return self.out_projection(out.view(B, L, -1))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_block, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = AttentionLayer(d_model, n_heads, attention_dropout=dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.conv4 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv6 = nn.Conv1d(d_model * 2, d_block, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu

    def forward(self, x):
        x_att = self.attention(x, x, x)
        x = x - self.dropout(x_att)
        x_ln = x = self.norm1(x)
        x_ln = self.dropout(self.act(self.conv1(x_ln.transpose(-1, 1))))
        x_ln = self.dropout(self.conv2(x_ln).transpose(-1, 1))
        x = (x - x_ln).transpose(-1, 1)
        h = self.conv4(x)
        out = self.conv6(torch.cat((x_att, x_ln), -1).transpose(-1, 1))
        return self.norm2(h.transpose(-1, 1)), out.transpose(-1, 1)


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, return_layer_outputs=False, return_backbone=False):
        output = 0
        layer_outputs = []
        for layer in self.layers:
            x, out = layer(x)
            output = output + out
            if return_layer_outputs:
                layer_outputs.append(out)
        if return_backbone:
            return x
        if return_layer_outputs:
            return output, layer_outputs
        return output
