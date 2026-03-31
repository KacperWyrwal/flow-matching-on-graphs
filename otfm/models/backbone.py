"""GNN backbone + transition scoring heads for configuration FM.

From config_fm/model.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """FiLM conditioning: scale and shift by global embedding."""

    def __init__(self, hidden_dim, global_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, h, global_emb):
        """h: (B, N, D), global_emb: (B, global_dim)."""
        film = self.proj(global_emb)
        gamma, beta = film.chunk(2, dim=-1)
        return h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class MessagePassingLayer(nn.Module):
    """Message passing with optional edge features."""

    def __init__(self, hidden_dim, edge_feat_dim=0):
        super().__init__()
        msg_in = hidden_dim * 2 + edge_feat_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.has_edge_feat = edge_feat_dim > 0

    def forward(self, h, edge_index, edge_feat=None):
        """h: (B, N, D), edge_index: (2, E), edge_feat: (E, d_e) or None."""
        B, N, D = h.shape
        src, dst = edge_index
        E = src.shape[0]

        h_src = h[:, src, :]  # (B, E, D)
        h_dst = h[:, dst, :]  # (B, E, D)

        if self.has_edge_feat and edge_feat is not None:
            if edge_feat.dim() == 2:
                # Static: (E, d_e) → tile across batch
                ef = edge_feat.unsqueeze(0).expand(B, E, -1)
            else:
                # Already batched: (B, E, d_e)
                ef = edge_feat
            msg_input = torch.cat([h_src, h_dst, ef], dim=-1)
        else:
            msg_input = torch.cat([h_src, h_dst], dim=-1)

        messages = self.msg_mlp(msg_input)  # (B, E, D)

        # Scatter-add to destination nodes
        agg = torch.zeros(B, N, D, device=h.device)
        dst_exp = dst.unsqueeze(0).unsqueeze(-1).expand(B, E, D)
        agg.scatter_add_(1, dst_exp, messages)

        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class FiLMGNNBackbone(nn.Module):
    """GNN backbone with FiLM conditioning for global features."""

    def __init__(self, node_feature_dim, edge_feature_dim=0,
                 global_dim=2, hidden_dim=128, n_layers=4):
        super().__init__()
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_feature_dim)
            for _ in range(n_layers)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, node_features, edge_index, edge_features,
                global_features):
        """
        node_features: (B, N, d_node)
        edge_index: (2, E)
        edge_features: (E, d_edge) or (B, E, d_edge) or None
        global_features: (B, d_global)
        Returns: (B, N, hidden_dim)
        """
        h = self.node_proj(node_features)
        global_emb = self.global_encoder(global_features)

        for mp, film in zip(self.mp_layers, self.film_layers):
            h = mp(h, edge_index, edge_features)
            h = film(h, global_emb)

        return h
