"""Adapted from SLA MAT SelfAttention / EncodeBlock (MIT).

Copyright (c) 2021 MAPPO contributors. See MAPPO_LICENSE and SOURCES.md.
Adaptations: one authorized seat's variable-length slots, key/query padding
masks, unequal key/query lengths, no joint-agent or autoregressive decoder.
"""

import math

import torch
from torch import nn


def initialized_linear(inputs, outputs, gain=.01):
    layer = nn.Linear(inputs, outputs)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class SlotAttention(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must divide into heads")
        self.heads = heads
        self.key = initialized_linear(width, width)
        self.query = initialized_linear(width, width)
        self.value = initialized_linear(width, width)
        self.proj = initialized_linear(width, width)

    def forward(self, key, value, query, key_mask, query_mask, attention_bias=None):
        batch, length, width = query.shape
        key_length = key.shape[1]
        if (key_mask.dtype != torch.bool or query_mask.dtype != torch.bool
                or key_mask.shape != (batch, key_length)
                or query_mask.shape != (batch, length)
                or not bool(key_mask.any(dim=1).all())):
            raise ValueError("Invalid or empty attention mask")
        if any(not bool(torch.isfinite(tensor[mask]).all())
               for tensor, mask in ((key, key_mask), (value, key_mask), (query, query_mask))):
            raise ValueError("Nonfinite real attention slot")
        key = key.masked_fill(~key_mask.unsqueeze(-1), 0.)
        value = value.masked_fill(~key_mask.unsqueeze(-1), 0.)
        query = query.masked_fill(~query_mask.unsqueeze(-1), 0.)
        k = self.key(key).view(batch, key_length, self.heads, width // self.heads).transpose(1, 2)
        q = self.query(query).view(batch, length, self.heads, width // self.heads).transpose(1, 2)
        v = self.value(value).view(batch, key_length, self.heads, width // self.heads).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(width // self.heads)
        if attention_bias is not None:
            if (attention_bias.shape != (batch, self.heads, length, key_length)
                    or not bool(torch.isfinite(attention_bias).all())):
                raise ValueError("Invalid typed relation attention bias")
            scores = scores + attention_bias
        scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attended = torch.softmax(scores, dim=-1) @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(attended) * query_mask.unsqueeze(-1)


class SlotBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = SlotAttention(width, heads)
        self.mlp = nn.Sequential(initialized_linear(width, width, math.sqrt(2)),
                                 nn.GELU(), initialized_linear(width, width))

    def forward(self, inputs, mask, attention_bias=None):
        inputs = inputs.masked_fill(~mask.unsqueeze(-1), 0.)
        values = self.ln1(inputs + self.attn(inputs, inputs, inputs, mask, mask, attention_bias))
        values = self.ln2(values + self.mlp(values))
        return values * mask.unsqueeze(-1)
