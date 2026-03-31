"""Scoring heads for configuration flow matching models.

From config_fm/model.py (SingleNodeHead, PairwiseAttentionHead)
and an EdgeRateHead alias for the edge MLP used in meta_fm/model.py predictors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleNodeHead(nn.Module):
    """k=1: per-node rate (DFM-style)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, edge_index):
        """h: (B, N, D) -> (B, N) raw scores."""
        return self.mlp(h).squeeze(-1)


class PairwiseAttentionHead(nn.Module):
    """k=2, binary vocab: pairwise swap rate via edge scoring.

    Computes scores only for edges in the position graph (sparse),
    then scatters into (B, N, N) output. Avoids O(N^2) dense expansion.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, edge_index):
        """h: (B, N, D), edge_index: (2, E) -> (B, N, N) raw scores."""
        B, N, D = h.shape
        src, dst = edge_index  # (E,)
        E = src.shape[0]

        h_src = h[:, src, :]  # (B, E, D)
        h_dst = h[:, dst, :]  # (B, E, D)
        edge_scores = self.edge_mlp(
            torch.cat([h_src, h_dst], dim=-1)).squeeze(-1)  # (B, E)

        # Scatter into dense (B, N, N) -- only position graph edges are nonzero
        out = torch.zeros(B, N, N, device=h.device)
        out[:, src, dst] = edge_scores
        return out


class SwapScoringHead(nn.Module):
    """k=4: score double edge swaps from node embeddings.

    Fully batched: pads to max swaps, builds index tensor, does one
    gather + one MLP call for the entire batch.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.swap_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h, valid_swaps_batch):
        """
        h: (B, n, d) node embeddings
        valid_swaps_batch: list of B lists, each containing
            (a, b, c, d, rewiring) tuples
        Returns: list of B tensors, each (n_swaps_b,)
        """
        B, N, D = h.shape
        swap_counts = [len(s) for s in valid_swaps_batch]
        max_swaps = max(swap_counts) if swap_counts else 0

        if max_swaps == 0:
            return [torch.zeros(0, device=h.device) for _ in range(B)]

        # Build padded index tensor: (B, max_swaps, 4)
        indices = torch.zeros(B, max_swaps, 4, dtype=torch.long,
                              device=h.device)
        for b_idx in range(B):
            for s_idx, (a, b, c, d, rewiring) in enumerate(
                    valid_swaps_batch[b_idx]):
                if rewiring == 'ac_bd':
                    indices[b_idx, s_idx, 0] = a
                    indices[b_idx, s_idx, 1] = b
                    indices[b_idx, s_idx, 2] = c
                    indices[b_idx, s_idx, 3] = d
                else:
                    indices[b_idx, s_idx, 0] = a
                    indices[b_idx, s_idx, 1] = b
                    indices[b_idx, s_idx, 2] = d
                    indices[b_idx, s_idx, 3] = c

        # Gather: (B, max_swaps, 4, D)
        batch_idx = torch.arange(B, device=h.device
                                 ).unsqueeze(1).unsqueeze(2).expand(
                                     -1, max_swaps, 4)
        swap_emb = h[batch_idx, indices]  # (B, max_swaps, 4, D)
        swap_feats = swap_emb.reshape(B, max_swaps, 4 * D)

        # Single MLP call
        all_scores = self.swap_mlp(swap_feats).squeeze(-1)  # (B, max_swaps)
        all_rates = F.softplus(all_scores)

        # Unpad
        result = []
        for b_idx in range(B):
            result.append(all_rates[b_idx, :swap_counts[b_idx]])
        return result


class EdgeRateHead(nn.Module):
    """Edge-level rate head: MLP on pairs of endpoint features -> softplus -> rate.

    Used by the GNN predictors in meta_fm/model.py for producing edge rates.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_src, h_dst):
        """
        h_src: (*, E, D)
        h_dst: (*, E, D)
        Returns: (*, E) softplus rates
        """
        edge_features = torch.cat([h_src, h_dst], dim=-1)
        return F.softplus(self.edge_mlp(edge_features).squeeze(-1))
