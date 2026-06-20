from __future__ import annotations

import torch
import torch.nn as nn

from nazari_vrp.data import VRPBatch


class NazariVRPModel(nn.Module):
    """RNN decoder with attention for CVRP.

    The model mirrors the structure described by Nazari et al. (2018):
    static node features are embedded once, dynamic demand/load features are
    embedded at each decoding step, and an RNN decoder state attends over
    feasible destinations.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.static_embed = nn.Linear(2, embed_dim)
        self.dynamic_embed = nn.Linear(2, embed_dim)
        self.decoder = nn.GRUCell(embed_dim, hidden_dim)

        self.attn_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_key = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

        self.critic = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_static(self, batch: VRPBatch) -> torch.Tensor:
        return self.static_embed(batch.coords_with_depot())

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def _dynamic_features(
        self,
        remaining: torch.Tensor,
        load: torch.Tensor,
        capacity: torch.Tensor,
    ) -> torch.Tensor:
        safe_capacity = capacity[:, None].clamp_min(1e-6)
        demand_ratio = remaining / safe_capacity
        load_ratio = load[:, None].expand_as(remaining) / safe_capacity
        return torch.stack((demand_ratio, load_ratio), dim=-1)

    def step_logits(
        self,
        static_emb: torch.Tensor,
        remaining: torch.Tensor,
        load: torch.Tensor,
        capacity: torch.Tensor,
        current_node: torch.Tensor,
        hidden: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked logits for the next destination."""
        batch_size, _, embed_dim = static_emb.shape
        gather_index = current_node.view(batch_size, 1, 1).expand(-1, 1, embed_dim)
        decoder_input = static_emb.gather(1, gather_index).squeeze(1)
        hidden = self.decoder(decoder_input, hidden)

        dynamic_emb = self.dynamic_embed(self._dynamic_features(remaining, load, capacity))
        keys = static_emb + dynamic_emb
        query = self.attn_query(hidden).unsqueeze(1)
        energy = torch.tanh(query + self.attn_key(keys))
        logits = self.attn_score(energy).squeeze(-1)
        logits = logits.masked_fill(~action_mask, -1e9)
        return logits, hidden

    def baseline_value(
        self,
        static_emb: torch.Tensor,
        remaining: torch.Tensor,
        capacity: torch.Tensor,
    ) -> torch.Tensor:
        full_load = capacity.clone()
        dynamic_emb = self.dynamic_embed(self._dynamic_features(remaining, full_load, capacity))
        pooled = (static_emb + dynamic_emb).mean(dim=1)
        return self.critic(pooled).squeeze(-1)
