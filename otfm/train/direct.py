"""
Training loop for DirectGNNPredictor.

From meta_fm/train.py::train_direct_gnn.
"""

import numpy as np
import torch

from otfm.core.utils import EMA


def train_direct_gnn(model, train_pairs, n_epochs=1000, lr=5e-4, device='cpu', seed=0,
                     ema_decay=0.999, batch_size=256, checkpoint_path=None):
    """
    Train DirectGNNPredictor with KL divergence loss: KL(true || predicted).

    train_pairs: list of (context_np, mu_source_np, edge_index) tuples where
        context_np:   (N, context_dim) numpy array
        mu_source_np: (N,) numpy array
        edge_index:   (2, E) torch.LongTensor

    Samples are grouped by graph topology (edge_index) and processed in
    batched forward passes using tiled edge_index, same strategy as
    train_flexible_conditional.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    ema = EMA(model, decay=ema_decay)
    rng = np.random.default_rng(seed)
    losses = []

    # Pre-convert to tensors and group by topology
    pairs_t = [
        (torch.tensor(ctx, dtype=torch.float32),
         torch.tensor(mu, dtype=torch.float32),
         ei)
        for ctx, mu, ei in train_pairs
    ]

    for epoch in range(n_epochs):
        idx = rng.permutation(len(pairs_t))
        epoch_loss = 0.0
        n_samples = 0

        # Process in batches
        for batch_start in range(0, len(idx), batch_size):
            batch_idx = idx[batch_start:batch_start + batch_size]

            # Group by topology (edge_index identity)
            groups: dict = {}
            for i in batch_idx:
                ctx_t, mu_t, ei = pairs_t[int(i)]
                key = ei.data_ptr()
                if key not in groups:
                    groups[key] = []
                groups[key].append((ctx_t, mu_t, ei))

            batch_loss = 0.0
            for group in groups.values():
                ei = group[0][2].to(device)
                N = group[0][0].shape[0]
                B = len(group)

                # Stack into batched tensors
                ctx_b = torch.stack([g[0] for g in group]).to(device)    # (B, N, C)
                mu_b = torch.stack([g[1] for g in group]).to(device)     # (B, N)

                # Tile edge_index for batched message passing
                src, dst = ei
                offsets = torch.arange(B, device=device) * N
                src_b = (src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
                dst_b = (dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
                batch_ei = torch.stack([src_b, dst_b])  # (2, B*E)

                # Flatten context for batched forward
                h = ctx_b.reshape(B * N, -1)  # (B*N, C)
                for mp_layer in model.mp_layers:
                    h = mp_layer(h, batch_ei)
                    h = model.dropout(h)

                logits = model.readout(h).squeeze(-1)  # (B*N,)
                logits = logits.view(B, N)
                mu_pred = torch.softmax(logits, dim=-1)  # (B, N)

                # KL loss per sample, sum over batch
                loss = (mu_b * (mu_b.clamp(min=1e-10).log()
                                - mu_pred.clamp(min=1e-10).log())).sum()
                batch_loss = batch_loss + loss

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            epoch_loss += batch_loss.item()
            n_samples += len(batch_idx)

        avg_loss = epoch_loss / max(n_samples, 1)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{n_epochs} | KL Loss: {avg_loss:.6f}")

        if checkpoint_path and (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), checkpoint_path)

    ema.apply(model)
    return {'losses': losses, 'ema': ema}
