"""
All predictor classes for flow matching on graphs.

From meta_fm/model.py and config_fm/model.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from otfm.models.conditioning import (
    RateMessagePassing,
    EdgeAwareMessagePassing,
    FiLMRateMessagePassing,
)
from otfm.models.backbone import FiLMGNNBackbone
from otfm.models.heads import SingleNodeHead, PairwiseAttentionHead


class RateMatrixPredictor(nn.Module):
    """
    Predicts a rate matrix given a distribution and time.

    Architecture:
        Input: [mu (N dims), t (1 dim)] -> N+1 dims
        Hidden: n_layers hidden layers, each hidden_dim units, ReLU activation
        Output: N*N dims, reshaped to (N, N)

    Post-processing:
        1. Mask diagonal to 0
        2. Apply softplus to off-diagonal entries (ensures >= 0)
        3. Set diagonal = -sum of off-diagonal in each row

    Constructor args:
        n_nodes: int
        hidden_dim: int = 256
        n_layers: int = 3
    """

    def __init__(self, n_nodes: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        input_dim = n_nodes + 1
        output_dim = n_nodes * n_nodes

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Off-diagonal mask: True where we have off-diagonal entries
        mask = ~torch.eye(n_nodes, dtype=torch.bool)
        self.register_buffer('off_diag_mask', mask)

    def forward(self, mu: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        mu: (batch, N)
        t: (batch, 1)
        returns: (batch, N, N) valid rate matrices
        """
        x = torch.cat([mu, t], dim=1)  # (batch, N+1)
        out = self.network(x)  # (batch, N*N)
        R = out.view(-1, self.n_nodes, self.n_nodes)  # (batch, N, N)

        # Zero the diagonal
        diag_mask = torch.eye(self.n_nodes, dtype=torch.bool, device=R.device)
        R = R.masked_fill(diag_mask.unsqueeze(0), 0.0)

        # Apply softplus to off-diagonal entries
        R_offdiag = F.softplus(R) * (~diag_mask).float().unsqueeze(0)

        # Set diagonal = -sum of off-diagonal in each row
        row_sums = R_offdiag.sum(dim=2)  # (batch, N)
        R_final = R_offdiag - torch.diag_embed(row_sums)

        return R_final


# Rename MLP version for reference
RateMatrixPredictorMLP = RateMatrixPredictor


class GNNRateMatrixPredictor(nn.Module):
    """
    Graph neural network that predicts rate matrices from (distribution, time).

    Architecture:
        Input: per-node features h_a = [mu(a), t]  (2 dims per node)
        Message passing: n_layers of RateMessagePassing
        Edge readout: MLP on pairs of endpoint features -> softplus -> rate
        Assembly: fill rate matrix on edges, set diagonal = -row sum

    Constructor args:
        edge_index: torch.LongTensor (2, num_edges)
        n_nodes: int
        hidden_dim: int = 64
        n_layers: int = 4
    """

    def __init__(self, edge_index: torch.LongTensor, n_nodes: int,
                 hidden_dim: int = 64, n_layers: int = 4):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.register_buffer('edge_index', edge_index)

        self.mp_layers = nn.ModuleList()
        self.mp_layers.append(RateMessagePassing(in_dim=2, hidden_dim=hidden_dim))
        for _ in range(n_layers - 1):
            self.mp_layers.append(RateMessagePassing(in_dim=hidden_dim, hidden_dim=hidden_dim))

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mu: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        mu: (batch, N)
        t: (batch, 1)
        returns: (batch, N, N) valid rate matrices
        """
        B, N = mu.shape
        device = mu.device

        t_expanded = t.expand(B, N)                           # (B, N)
        node_features = torch.stack([mu, t_expanded], dim=-1) # (B, N, 2)

        src, dst = self.edge_index
        offsets = torch.arange(B, device=device) * N
        src_b = (src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst_b = (dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        batch_ei = torch.stack([src_b, dst_b])

        h = node_features.reshape(B * N, -1)  # (B*N, 2)
        for mp_layer in self.mp_layers:
            h = mp_layer(h, batch_ei)  # (B*N, hidden_dim)

        h = h.view(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        src, dst = self.edge_index  # unbatched edge indices
        h_src = h[:, src, :]  # (B, num_edges, hidden_dim)
        h_dst = h[:, dst, :]  # (B, num_edges, hidden_dim)
        edge_features = torch.cat([h_src, h_dst], dim=-1)       # (B, num_edges, 2*hidden_dim)
        edge_rates = F.softplus(self.edge_mlp(edge_features).squeeze(-1))  # (B, num_edges)

        rate_matrix = torch.zeros(B, N, N, device=device)
        rate_matrix[:, src, dst] = edge_rates
        rate_matrix[:, range(N), range(N)] = -rate_matrix.sum(dim=-1)

        return rate_matrix


class ConditionalGNNRateMatrixPredictor(nn.Module):
    """
    GNN rate matrix predictor conditioned on additional per-node context.

    Input per node: [mu(a), t, context_1(a), ..., context_K(a)]
    = 2 + context_dim features.

    Architecture is identical to GNNRateMatrixPredictor except the first
    message-passing layer takes in_dim = 2 + context_dim.

    Constructor args:
        edge_index: torch.LongTensor (2, num_edges)
        n_nodes: int
        context_dim: int -- number of additional context features per node
        hidden_dim: int = 64
        n_layers: int = 4

    forward(mu, t, context):
        mu: (batch, N)
        t:  (batch, 1)
        context: (batch, N, context_dim)
        returns: (batch, N, N) valid rate matrices
    """

    def __init__(self, edge_index: torch.LongTensor, n_nodes: int,
                 context_dim: int = 2, hidden_dim: int = 64, n_layers: int = 4):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.register_buffer('edge_index', edge_index)

        in_dim_first = 2 + context_dim
        self.mp_layers = nn.ModuleList()
        self.mp_layers.append(RateMessagePassing(in_dim=in_dim_first, hidden_dim=hidden_dim))
        for _ in range(n_layers - 1):
            self.mp_layers.append(RateMessagePassing(in_dim=hidden_dim, hidden_dim=hidden_dim))

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mu: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        mu:      (batch, N)
        t:       (batch, 1)
        context: (batch, N, context_dim)
        returns: (batch, N, N)
        """
        B, N = mu.shape
        device = mu.device

        t_expanded = t.expand(B, N)                             # (B, N)
        base_feats = torch.stack([mu, t_expanded], dim=-1)      # (B, N, 2)
        node_features = torch.cat([base_feats, context], dim=-1)  # (B, N, 2+context_dim)

        src, dst = self.edge_index
        offsets = torch.arange(B, device=device) * N
        src_b = (src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst_b = (dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        batch_ei = torch.stack([src_b, dst_b])

        h = node_features.reshape(B * N, -1)  # (B*N, 2+context_dim)
        for mp_layer in self.mp_layers:
            h = mp_layer(h, batch_ei)

        h = h.view(B, N, self.hidden_dim)

        src, dst = self.edge_index
        h_src = h[:, src, :]
        h_dst = h[:, dst, :]
        edge_features = torch.cat([h_src, h_dst], dim=-1)
        edge_rates = F.softplus(self.edge_mlp(edge_features).squeeze(-1))

        rate_matrix = torch.zeros(B, N, N, device=device)
        rate_matrix[:, src, dst] = edge_rates
        rate_matrix[:, range(N), range(N)] = -rate_matrix.sum(dim=-1)

        return rate_matrix


class FlexibleConditionalGNNRateMatrixPredictor(nn.Module):
    """
    Like ConditionalGNNRateMatrixPredictor but accepts edge_index as input
    rather than storing it as a fixed buffer. Enables varying graph topology
    per forward pass. Processes one graph at a time (loop approach).

    Constructor args:
        context_dim: int = 2
        hidden_dim:  int = 64
        n_layers:    int = 4
        (no edge_index or n_nodes -- these vary per input)

    forward_single(mu, t, context, edge_index):
        mu:         (N,)              node distribution
        t:          (1,)              flow time
        context:    (N, context_dim)  per-node conditioning
        edge_index: (2, E)            graph edges
        returns:    (N, N)            u_tilde = (1-t)*R rate matrix
    """

    def __init__(self, context_dim: int = 2, hidden_dim: int = 64,
                 n_layers: int = 4, edge_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.edge_dim = edge_dim

        in_dim_first = 2 + context_dim
        self.mp_layers = nn.ModuleList()
        if edge_dim > 0:
            self.mp_layers.append(EdgeAwareMessagePassing(
                in_dim=in_dim_first, hidden_dim=hidden_dim, edge_dim=edge_dim))
            for _ in range(n_layers - 1):
                self.mp_layers.append(EdgeAwareMessagePassing(
                    in_dim=hidden_dim, hidden_dim=hidden_dim, edge_dim=edge_dim))
        else:
            self.mp_layers.append(RateMessagePassing(
                in_dim=in_dim_first, hidden_dim=hidden_dim))
            for _ in range(n_layers - 1):
                self.mp_layers.append(RateMessagePassing(
                    in_dim=hidden_dim, hidden_dim=hidden_dim))

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_single(self, mu: torch.Tensor, t: torch.Tensor,
                       context: torch.Tensor,
                       edge_index: torch.LongTensor,
                       edge_feat: torch.Tensor = None) -> torch.Tensor:
        """
        mu:         (N,)
        t:          (1,) tensor
        context:    (N, context_dim)
        edge_index: (2, E)
        edge_feat:  (E, edge_dim) or None
        Returns:    (N, N) u_tilde = (1-t)*R rate matrix
        """
        N = mu.shape[0]
        device = mu.device

        t_expanded = t.view(1).expand(N)                          # (N,)
        base_feats = torch.stack([mu, t_expanded], dim=-1)        # (N, 2)
        if self.context_dim > 0:
            node_features = torch.cat([base_feats, context], dim=-1)
        else:
            node_features = base_feats                            # (N, 2)

        h = node_features
        for mp_layer in self.mp_layers:
            if self.edge_dim > 0:
                h = mp_layer(h, edge_index, edge_feat)
            else:
                h = mp_layer(h, edge_index)                       # (N, hidden_dim)

        src, dst = edge_index
        edge_features = torch.cat([h[src], h[dst]], dim=-1)       # (E, 2*hidden_dim)
        edge_rates = F.softplus(self.edge_mlp(edge_features).squeeze(-1))  # (E,)

        rate_matrix = torch.zeros(N, N, device=device)
        rate_matrix[src, dst] = edge_rates
        arange_N = torch.arange(N, device=device)
        rate_matrix[arange_N, arange_N] = -rate_matrix.sum(dim=-1)

        return rate_matrix

    def forward_batch(self, mu: torch.Tensor, t: torch.Tensor,
                      context: torch.Tensor,
                      edge_index: torch.LongTensor,
                      edge_feat: torch.Tensor = None) -> torch.Tensor:
        """
        Batched forward for samples that share the same graph topology.

        mu:         (B, N)
        t:          (B, 1)
        context:    (B, N, context_dim)
        edge_index: (2, E) -- same topology for all B samples
        edge_feat:  (E, edge_dim) or None -- same for all B samples
        Returns:    (B, N, N) u_tilde rate matrices
        """
        B, N = mu.shape
        device = mu.device

        t_expanded = t.expand(B, N)                               # (B, N)
        base_feats = torch.stack([mu, t_expanded], dim=-1)        # (B, N, 2)
        if self.context_dim > 0:
            node_features = torch.cat([base_feats, context], dim=-1)
        else:
            node_features = base_feats                            # (B, N, 2)

        # Tile edge_index across batch: offset by N*b for sample b
        src, dst = edge_index
        E = src.shape[0]
        offsets = torch.arange(B, device=device) * N              # (B,)
        src_b = (src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)  # (B*E,)
        dst_b = (dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        batch_ei = torch.stack([src_b, dst_b])                    # (2, B*E)

        # Tile edge features across batch
        ef_b = None
        if self.edge_dim > 0 and edge_feat is not None:
            ef_b = edge_feat.unsqueeze(0).expand(B, -1, -1).reshape(B * E, -1)

        h = node_features.reshape(B * N, -1)                      # (B*N, feat)
        for mp_layer in self.mp_layers:
            if self.edge_dim > 0:
                h = mp_layer(h, batch_ei, ef_b)
            else:
                h = mp_layer(h, batch_ei)                         # (B*N, hidden)

        h = h.view(B, N, self.hidden_dim)

        h_src = h[:, src, :]                                      # (B, E, hidden)
        h_dst = h[:, dst, :]
        edge_features = torch.cat([h_src, h_dst], dim=-1)         # (B, E, 2*hidden)
        edge_rates = F.softplus(self.edge_mlp(edge_features).squeeze(-1))  # (B, E)

        rate_matrix = torch.zeros(B, N, N, device=device)
        rate_matrix[:, src, dst] = edge_rates
        arange_N = torch.arange(N, device=device)
        rate_matrix[:, arange_N, arange_N] = -rate_matrix.sum(dim=-1)

        return rate_matrix


class FiLMConditionalGNNRateMatrixPredictor(nn.Module):
    """
    GNN with dual conditioning:
      - Per-node context features (sensor values at sensor locations + binary mask)
      - Global FiLM conditioning from raw sensor vector y + tau_diff

    Constructor args:
        node_context_dim: int = 2  (sensor_value * is_sensor, is_sensor)
        global_dim: int = 21  (raw sensor vector + tau_diff)
        hidden_dim: int = 128
        n_layers: int = 6

    forward_single(mu, t, node_context, global_cond, edge_index):
        mu:           (N,)
        t:            (1,) tensor
        node_context: (N, node_context_dim)
        global_cond:  (global_dim,)
        edge_index:   (2, E)
        returns:      (N, N) u_tilde = (1-t)*R rate matrix

    forward_batch(mu, t, node_context, global_cond, edge_index):
        mu:           (B, N)
        t:            (B, 1)
        node_context: (B, N, node_context_dim)
        global_cond:  (B, global_dim)
        edge_index:   (2, E) same topology for all B samples
        returns:      (B, N, N) u_tilde rate matrices
    """

    def __init__(self, node_context_dim: int = 2, global_dim: int = 21,
                 hidden_dim: int = 128, n_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_context_dim = node_context_dim

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        input_dim = 2 + node_context_dim
        self.mp_layers = nn.ModuleList()
        self.mp_layers.append(FiLMRateMessagePassing(input_dim, hidden_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.mp_layers.append(FiLMRateMessagePassing(hidden_dim, hidden_dim, hidden_dim))

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_single(self, mu: torch.Tensor, t: torch.Tensor,
                       node_context: torch.Tensor, global_cond: torch.Tensor,
                       edge_index: torch.LongTensor) -> torch.Tensor:
        N = mu.shape[0]
        device = mu.device

        t_expanded = t.view(1).expand(N)
        h = torch.cat([mu.unsqueeze(-1), t_expanded.unsqueeze(-1),
                       node_context], dim=-1)  # (N, 2+ctx)

        g = self.global_encoder(global_cond)  # (hidden_dim,) -- broadcasts to all nodes

        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, g)

        src, dst = edge_index
        edge_features = torch.cat([h[src], h[dst]], dim=-1)
        edge_rates = F.softplus(self.edge_mlp(edge_features).squeeze(-1))

        rate_matrix = torch.zeros(N, N, device=device)
        rate_matrix[src, dst] = edge_rates
        arange_N = torch.arange(N, device=device)
        rate_matrix[arange_N, arange_N] = -rate_matrix.sum(dim=-1)

        return rate_matrix

    def forward_batch(self, mu: torch.Tensor, t: torch.Tensor,
                      node_context: torch.Tensor, global_cond: torch.Tensor,
                      edge_index: torch.LongTensor) -> torch.Tensor:
        B, N = mu.shape
        device = mu.device

        t_expanded = t.expand(B, N)
        base_feats = torch.stack([mu, t_expanded], dim=-1)  # (B, N, 2)
        node_features = torch.cat([base_feats, node_context], dim=-1)  # (B, N, 2+ctx)

        src, dst = edge_index
        offsets = torch.arange(B, device=device) * N
        src_b = (src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst_b = (dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        batch_ei = torch.stack([src_b, dst_b])

        h = node_features.reshape(B * N, -1)  # (B*N, 2+ctx)

        g = self.global_encoder(global_cond)  # (B, hidden_dim)
        g_expanded = g.unsqueeze(1).expand(B, N, self.hidden_dim).reshape(B * N, self.hidden_dim)

        for mp_layer in self.mp_layers:
            h = mp_layer(h, batch_ei, g_expanded)  # (B*N, hidden_dim)

        h = h.view(B, N, self.hidden_dim)

        h_src = h[:, src, :]
        h_dst = h[:, dst, :]
        edge_features = torch.cat([h_src, h_dst], dim=-1)
        edge_rates = F.softplus(self.edge_mlp(edge_features).squeeze(-1))

        rate_matrix = torch.zeros(B, N, N, device=device)
        rate_matrix[:, src, dst] = edge_rates
        arange_N = torch.arange(N, device=device)
        rate_matrix[:, arange_N, arange_N] = -rate_matrix.sum(dim=-1)

        return rate_matrix


class DirectGNNPredictor(nn.Module):
    """
    Same message-passing backbone as FlexibleConditionalGNNRateMatrixPredictor,
    but the readout is per-node logits -> softmax -> distribution.

    One forward pass predicts the source distribution directly (no ODE integration).

    forward(context, edge_index):
        context:    (N, context_dim)
        edge_index: (2, E)
        returns:    (N,) distribution (sums to 1)
    """

    def __init__(self, context_dim: int = 2, hidden_dim: int = 128, n_layers: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.mp_layers = nn.ModuleList()
        self.mp_layers.append(RateMessagePassing(in_dim=context_dim, hidden_dim=hidden_dim))
        for _ in range(n_layers - 1):
            self.mp_layers.append(RateMessagePassing(in_dim=hidden_dim, hidden_dim=hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context, edge_index):
        h = context
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index)
            h = self.dropout(h)
        logits = self.readout(h).squeeze(-1)  # (N,)
        return torch.softmax(logits, dim=0)   # (N,) distribution


class EdgeAwareDirectGNNPredictor(nn.Module):
    """Direct prediction of target distribution using edge-aware GNN.

    No flow matching -- single forward pass from edge features to output.
    Input per node: constant (no context needed). All information comes
    from edge features through message passing.

    forward(edge_index, edge_feat, N):
        edge_index: (2, E) directed edges
        edge_feat:  (E, edge_dim) per-edge features
        N:          number of nodes
        returns:    (N,) distribution (sums to 1)
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 6,
                 edge_dim: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)

        self.mp_layers = nn.ModuleList([
            EdgeAwareMessagePassing(hidden_dim, hidden_dim, edge_dim)
            for _ in range(n_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_index, edge_feat, N):
        """
        edge_index: (2, E)
        edge_feat:  (E, edge_dim)
        N:          number of nodes
        Returns:    (N,) softmax distribution
        """
        h = torch.ones(N, 1, device=edge_feat.device)
        h = self.input_proj(h)

        for mp in self.mp_layers:
            h = mp(h, edge_index, edge_feat)

        logits = self.readout(h).squeeze(-1)  # (N,)
        return torch.softmax(logits, dim=0)


class ConfigurationRatePredictor(nn.Module):
    """Unified model for configuration flow matching.

    GNN backbone -> node embeddings -> scoring head -> masked rates.
    """

    def __init__(self, node_feature_dim, edge_feature_dim=0,
                 global_dim=2, hidden_dim=128, n_layers=4,
                 transition_order=2):
        super().__init__()
        self.transition_order = transition_order

        self.backbone = FiLMGNNBackbone(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        if transition_order == 1:
            self.head = SingleNodeHead(hidden_dim)
        elif transition_order == 2:
            self.head = PairwiseAttentionHead(hidden_dim)
        elif transition_order >= 4:
            from otfm.models.heads import SwapScoringHead
            self.head = SwapScoringHead(hidden_dim)
        else:
            raise ValueError(f"Unsupported transition_order={transition_order}")

    def forward(self, node_features, edge_index, edge_features,
                global_features, transition_mask):
        """
        For k=1, k=2: mask-based API.
        node_features: (B, N, d_node)
        edge_index: (2, E)
        edge_features: (E, d_edge) or None
        global_features: (B, d_global)
        transition_mask: (B, ...) valid transition mask
        Returns: (B, ...) predicted rates, zero where mask is zero
        """
        h = self.backbone(node_features, edge_index, edge_features,
                          global_features)
        raw_rates = self.head(h, edge_index)
        return F.softplus(raw_rates) * transition_mask

    def score_transitions(self, node_features, edge_index, edge_features,
                          global_features, valid_swaps_batch):
        """
        For k>=4: enumeration-based API.
        valid_swaps_batch: list of B lists of (a, b, c, d, rewiring)
        Returns: list of B tensors, each (n_swaps_b,) rates
        """
        h = self.backbone(node_features, edge_index, edge_features,
                          global_features)
        return self.head(h, valid_swaps_batch)
