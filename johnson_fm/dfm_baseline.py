"""DFM baseline: independent bit-flip rates on {0,1}^n (unconstrained)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from johnson_fm.model import MessagePassingLayer, FiLMLayer


class DFMBitFlipPredictor(nn.Module):
    """Per-position flip rate predictor for DFM baseline.

    Same architecture as SwapRatePredictor but output is per-position
    flip rates instead of pairwise swap rates. Receives J and h.
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

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t, t, beta, J_mat, h_field):
        """
        x_t: (B, n) binary configuration
        t: (B, 1) flow time
        beta: (B, 1) inverse temperature
        J_mat: (n, n) coupling matrix
        h_field: (n,) external field
        Returns: (B, n) flip rates (non-negative)
        """
        B, n = x_t.shape

        h_broadcast = h_field.unsqueeze(0).expand(B, n)
        node_feats = torch.stack([x_t, h_broadcast], dim=-1)  # (B, n, 2)
        h = self.node_proj(node_feats)

        edge_feat = J_mat.unsqueeze(0).unsqueeze(-1).expand(B, n, n, 1)

        global_raw = torch.cat([t, beta], dim=-1)
        global_emb = self.global_encoder(global_raw)

        for mp, film in zip(self.mp_layers, self.film_layers):
            h = mp(h, edge_feat)
            h = film(h, global_emb)

        rates = F.softplus(self.readout(h).squeeze(-1))  # (B, n)
        return rates
