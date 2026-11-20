"""Seat-visible relational slot actor and separate critic; lazy Torch import.

Only frozen per-slot text encodings are shared. No absolute player/slot
position embedding, other-seat private state or joint-team action decoder.
"""

ARCHITECTURE = {
    "version": "contextual-relational-slots-v1", "width": 128, "heads": 4,
    "blocks": 2, "dropout": 0., "text_dimension": 384,
    "numeric_dimension": 32, "slot_types": 7, "relation_types": 16,
    "actor_critic_shared_trainable_parameters": False,
}


def build_network():
    import torch
    from torch import nn
    from .vendor.slot_attention import SlotBlock

    class SlotBody(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_projection = nn.Linear(384, 128)
            self.numeric_projection = nn.ModuleList([nn.Linear(32, 128) for _ in range(7)])
            self.type_embedding = nn.Embedding(7, 128)
            self.blocks = nn.ModuleList([SlotBlock(128, 4) for _ in range(2)])
            self.relation_weights = nn.Parameter(torch.zeros(2, 4, 16))

        def forward(self, inputs):
            text, numeric = inputs["text"], inputs["numeric"]
            types, mask, relations = inputs["types"], inputs["mask"], inputs["relations"]
            batch, length, dimension = text.shape
            if (dimension != 384 or numeric.shape != (batch, length, 32)
                    or types.shape != mask.shape or mask.shape != (batch, length)
                    or relations.shape != (batch, length, length, 16)
                    or mask.dtype != torch.bool or types.dtype != torch.int64
                    or not bool(mask.any(dim=1).all())
                    or bool(((types < 0) | (types >= 7)).any())
                    or not bool(torch.isfinite(text[mask]).all())
                    or not bool(torch.isfinite(numeric[mask]).all())
                    or not bool(torch.isfinite(relations).all())
                    or bool(((relations < 0) | (relations > 1)).any())):
                raise ValueError("Invalid typed slot tensors")
            text = text.masked_fill(~mask.unsqueeze(-1), 0.)
            numeric = numeric.masked_fill(~mask.unsqueeze(-1), 0.)
            hidden = self.text_projection(text) + self.type_embedding(types)
            for index, projection in enumerate(self.numeric_projection):
                hidden = hidden + projection(numeric) * (types == index).unsqueeze(-1)
            hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.)
            pair_mask = mask[:, :, None, None] & mask[:, None, :, None]
            relations = relations.masked_fill(~pair_mask, 0.)
            for index, block in enumerate(self.blocks):
                bias = torch.einsum("bijr,hr->bhij", relations, self.relation_weights[index])
                hidden = block(hidden, mask, bias)
            return hidden

    class SlotActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = SlotBody()
            self.score = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
            nn.init.orthogonal_(self.score[-1].weight, gain=.01)
            nn.init.zeros_(self.score[-1].bias)

        def forward(self, inputs):
            hidden = self.body(inputs)
            indices, legal = inputs["candidate_indices"], inputs["candidate_mask"]
            batch, count = indices.shape
            if (count < 1 or count > 8 or indices.dtype != torch.int64
                    or legal.dtype != torch.bool or legal.shape != indices.shape
                    or batch != hidden.shape[0] or not bool(legal.any(dim=1).all())
                    or bool(((indices < 0) | (indices >= hidden.shape[1])).any())):
                raise ValueError("Invalid candidate slot mapping")
            for row in range(batch):
                actual = indices[row, legal[row]]
                if (len(actual.unique()) != len(actual)
                        or not bool(inputs["mask"][row, actual].all())
                        or not bool((inputs["types"][row, actual] == 5).all())):
                    raise ValueError("Candidates must map to unique live candidate slots")
            slots = torch.gather(hidden, 1, indices.unsqueeze(-1).expand(-1, -1, 128))
            return self.score(slots).squeeze(-1).masked_fill(~legal, 0.)

    class SlotCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = SlotBody()
            self.value = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))

        def forward(self, inputs):
            hidden = self.body(inputs)
            pooled = hidden.sum(dim=1) / inputs["mask"].sum(dim=1).unsqueeze(-1)
            return self.value(pooled).squeeze(-1)

    class ContextualNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor, self.critic = SlotActor(), SlotCritic()

        def forward(self, inputs):
            return self.actor(inputs), self.critic(inputs)

    return ContextualNetwork()
