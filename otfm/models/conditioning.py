"""Message passing layers with conditioning (FiLM, edge-aware).

From meta_fm/model.py: RateMessagePassing, FiLMRateMessagePassing,
EdgeAwareMessagePassing.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class RateMessagePassing(MessagePassing):
    """
    Custom message-passing layer for rate matrix prediction.

    Uses "source_to_target" flow convention (PyG default).

    message(x_i, x_j): MLP_msg(cat(x_i, x_j))
    update(aggr_out, x): MLP_update(cat(x, aggr_out))
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index):
        aggr = self.propagate(edge_index, x=x)
        return self.update_mlp(torch.cat([x, aggr], dim=-1))

    def message(self, x_i, x_j):
        return self.msg_mlp(torch.cat([x_i, x_j], dim=-1))


class EdgeAwareMessagePassing(nn.Module):
    """Message passing with optional edge features.

    If edge_feat is None, behaves identically to RateMessagePassing.
    If edge_feat is provided (E, edge_dim), concatenates it to the message input.
    """

    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int = 0):
        super().__init__()
        msg_input_dim = 2 * in_dim + edge_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual = (nn.Linear(in_dim, hidden_dim)
                         if in_dim != hidden_dim else nn.Identity())

    def forward(self, h, edge_index, edge_feat=None):
        """
        h:          (N, in_dim) or (B*N, in_dim) node features
        edge_index: (2, E) or (2, B*E) directed edges
        edge_feat:  (E, edge_dim) or (B*E, edge_dim) or None
        """
        src, dst = edge_index

        if edge_feat is not None:
            msg_input = torch.cat([h[src], h[dst], edge_feat], dim=-1)
        else:
            msg_input = torch.cat([h[src], h[dst]], dim=-1)

        messages = self.msg_mlp(msg_input)                           # (E, hidden)

        N = h.shape[0]
        agg = torch.zeros(N, messages.shape[1], device=h.device)
        agg.index_add_(0, dst, messages)

        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        h_new = h_new + self.residual(h)
        return h_new


class FiLMRateMessagePassing(MessagePassing):
    """
    Message passing layer with FiLM conditioning from a global encoded vector.

    After the standard message + update step, modulate node features:
        h_a <- gamma * h_a + beta
    where (gamma, beta) = film_mlp(global_cond).

    global_cond can be:
        - (global_dim,) for single-sample: broadcasts to all N nodes
        - (B*N, global_dim) for batch: per-node conditioning (already expanded)
    """

    def __init__(self, in_dim: int, hidden_dim: int, global_dim: int):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

        # Identity initialization: gamma=1, beta=0 so FiLM starts as pass-through
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)
        self.film_mlp[-1].bias.data[:hidden_dim] = 1.0  # gamma = 1

    def forward(self, x, edge_index, global_cond):
        aggr = self.propagate(edge_index, x=x)
        h = self.update_mlp(torch.cat([x, aggr], dim=-1))
        film_params = self.film_mlp(global_cond)
        gamma, beta = film_params.chunk(2, dim=-1)
        h = gamma * h + beta
        return h

    def message(self, x_i, x_j):
        return self.msg_mlp(torch.cat([x_i, x_j], dim=-1))
