"""GNN swap rate predictor for Johnson graph flow matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """FiLM conditioning: scale and shift node features by global embedding."""

    def __init__(self, hidden_dim, global_dim):
        super().__init__()
        self.film_proj = nn.Sequential(
            nn.Linear(global_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, h, global_emb):
        """h: (B, N, D), global_emb: (B, global_dim) -> (B, N, D)."""
        film = self.film_proj(global_emb)  # (B, 2D)
        gamma, beta = film.chunk(2, dim=-1)  # each (B, D)
        return h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class MessagePassingLayer(nn.Module):
    """Message passing on complete graph with edge features (J_ij)."""

    def __init__(self, hidden_dim, edge_feat_dim=1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_feat):
        """h: (B, N, D), edge_feat: (B, N, N, edge_feat_dim) -> (B, N, D)."""
        B, N, D = h.shape
        h_i = h.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)
        msg_input = torch.cat([h_i, h_j, edge_feat], dim=-1)  # (B,N,N,2D+ef)
        msg = self.msg_mlp(msg_input)  # (B, N, N, D)
        agg = msg.sum(dim=2)  # (B, N, D)
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))  # (B, N, D)
        return self.norm(h + h_new)


class SwapRatePredictor(nn.Module):
    """GNN on n position nodes, predicts swap rates for valid swaps.

    Node features: [x_t[i], h[i]] — current value + external field
    Edge features: J[i,j] — coupling matrix (provided to message passing)
    Global features: [t, beta] via FiLM conditioning
    Output: (B, n, n) swap rates masked to valid pairs (x_t[i]=1, x_t[j]=0)
    """

    def __init__(self, node_feature_dim=2, global_dim=2,
                 hidden_dim=128, n_layers=4, edge_feat_dim=1):
        super().__init__()
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)

        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_feat_dim)
            for _ in range(n_layers)
        ])
        self.film_layers = nn.ModuleList([
            FiLMLayer(hidden_dim, global_dim) for _ in range(n_layers)
        ])

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, global_dim),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t, t, beta, J_mat, h_field):
        """
        x_t: (B, n) binary configuration
        t: (B, 1) flow time
        beta: (B, 1) inverse temperature
        J_mat: (n, n) coupling matrix (shared across batch)
        h_field: (n,) external field (shared across batch)
        Returns: (B, n, n) swap rates, masked to valid pairs
        """
        B, n = x_t.shape
        device = x_t.device

        # Node features: [x_t[i], h[i]]
        h_broadcast = h_field.unsqueeze(0).expand(B, n)  # (B, n)
        node_feats = torch.stack([x_t, h_broadcast], dim=-1)  # (B, n, 2)
        h = self.node_proj(node_feats)  # (B, n, D)

        # Edge features: J[i,j] (shared across batch)
        edge_feat = J_mat.unsqueeze(0).unsqueeze(-1).expand(
            B, n, n, 1)  # (B, n, n, 1)

        # Global conditioning
        global_raw = torch.cat([t, beta], dim=-1)  # (B, 2)
        global_emb = self.global_encoder(global_raw)  # (B, global_dim)

        # Message passing + FiLM
        for mp, film in zip(self.mp_layers, self.film_layers):
            h = mp(h, edge_feat)
            h = film(h, global_emb)

        # Edge-level rate prediction: include J_ij in edge readout
        h_i = h.unsqueeze(2).expand(B, n, n, -1)  # (B, n, n, D)
        h_j = h.unsqueeze(1).expand(B, n, n, -1)  # (B, n, n, D)
        edge_input = torch.cat([h_i, h_j, edge_feat], dim=-1)
        raw_rates = self.edge_mlp(edge_input).squeeze(-1)  # (B, n, n)
        rates = F.softplus(raw_rates)

        # Mask: only valid swaps (i has 1, j has 0)
        valid_mask = x_t.unsqueeze(-1) * (1 - x_t).unsqueeze(-2)  # (B, n, n)
        return rates * valid_mask
