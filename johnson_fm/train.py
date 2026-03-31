"""Training loops for Johnson graph flow matching and DFM baseline."""

import numpy as np
import torch
import torch.nn.functional as F

from johnson_fm.energy import ising_energy, uniform_sample
from johnson_fm.flow import sample_intermediate, compute_target_rates


def train_swap_fm(model, J, h, n, k, betas, mcmc_pools,
                  n_epochs=2000, batch_size=256, lr=5e-4,
                  device='cpu', seed=42):
    """Train SwapRatePredictor with on-the-fly data generation.

    J and h are passed to the model as edge/node features so it can learn
    the energy landscape.
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    rng = np.random.default_rng(seed)
    losses = []

    # Pre-compute J and h tensors (shared across all batches)
    J_t = torch.tensor(J, dtype=torch.float32, device=device)
    h_t = torch.tensor(h, dtype=torch.float32, device=device)

    steps_per_epoch = max(1, 1000 // batch_size)

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            x_t_batch = []
            t_batch = []
            beta_batch = []
            target_batch = []

            for _ in range(batch_size):
                beta_val = float(rng.choice(betas))
                pool = mcmc_pools[beta_val]
                x_T = pool[int(rng.integers(len(pool)))].copy()
                x_0 = uniform_sample(n, k, rng)
                t = float(rng.uniform(0.0, 0.999))

                x_t, S_plus_rem, S_minus_rem, d, ell = sample_intermediate(
                    x_0, x_T, t, rng)
                target_rates = compute_target_rates(
                    n, S_plus_rem, S_minus_rem, d, ell)

                x_t_batch.append(x_t)
                t_batch.append(t)
                beta_batch.append(beta_val)
                target_batch.append(target_rates)

            x_t_t = torch.tensor(np.array(x_t_batch),
                                 dtype=torch.float32, device=device)
            t_t = torch.tensor(np.array(t_batch),
                               dtype=torch.float32, device=device).unsqueeze(-1)
            beta_t = torch.tensor(np.array(beta_batch),
                                  dtype=torch.float32, device=device).unsqueeze(-1)
            target_t = torch.tensor(np.array(target_batch),
                                    dtype=torch.float32, device=device)

            # Forward with J and h
            pred_rates = model(x_t_t, t_t, beta_t, J_t, h_t)  # (B, n, n)

            # Rate KL loss over valid swap pairs
            mask = target_t > 0
            if mask.any():
                r_true = target_t[mask]
                r_pred = pred_rates[mask].clamp(min=1e-10)
                loss_kl = (r_true * (r_true.clamp(min=1e-10).log()
                                     - r_pred.log())
                           - r_true + r_pred).sum()
                # Penalize nonzero predictions where target is 0
                mask_zero = (target_t == 0) & (pred_rates > 0)
                if mask_zero.any():
                    loss_kl = loss_kl + pred_rates[mask_zero].sum() * 0.1
                loss = loss_kl / batch_size
            else:
                loss = pred_rates.sum() * 0.0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f}",
                  flush=True)

    return {'losses': losses}


def train_dfm(model, J, h, n, k, betas, mcmc_pools,
              n_epochs=2000, batch_size=256, lr=5e-4,
              device='cpu', seed=42):
    """Train DFMBitFlipPredictor with per-position flip rates.

    J and h are passed to the model.
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    rng = np.random.default_rng(seed)
    losses = []

    J_t = torch.tensor(J, dtype=torch.float32, device=device)
    h_t = torch.tensor(h, dtype=torch.float32, device=device)

    steps_per_epoch = max(1, 1000 // batch_size)

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            x_t_batch = []
            t_batch = []
            beta_batch = []
            target_batch = []

            for _ in range(batch_size):
                beta_val = float(rng.choice(betas))
                pool = mcmc_pools[beta_val]
                x_T = pool[int(rng.integers(len(pool)))].copy()

                # Source: uniform over {0,1}^n (no k constraint)
                x_0 = rng.binomial(1, 0.5, size=n).astype(np.float32)

                t = float(rng.uniform(0.0, 0.999))

                # Per-position independent interpolation
                switch = rng.uniform(size=n) < t
                x_t = np.where(switch, x_T, x_0)

                # Target flip rate
                target_rate = np.zeros(n, dtype=np.float32)
                diff = x_t != x_T
                if diff.any():
                    target_rate[diff] = 1.0 / (1.0 - t + 1e-10)

                x_t_batch.append(x_t)
                t_batch.append(t)
                beta_batch.append(beta_val)
                target_batch.append(target_rate)

            x_t_t = torch.tensor(np.array(x_t_batch),
                                 dtype=torch.float32, device=device)
            t_t = torch.tensor(np.array(t_batch),
                               dtype=torch.float32, device=device).unsqueeze(-1)
            beta_t = torch.tensor(np.array(beta_batch),
                                  dtype=torch.float32, device=device).unsqueeze(-1)
            target_t = torch.tensor(np.array(target_batch),
                                    dtype=torch.float32, device=device)

            pred_rates = model(x_t_t, t_t, beta_t, J_t, h_t)  # (B, n)

            loss = F.mse_loss(pred_rates, target_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  DFM Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f}",
                  flush=True)

    return {'losses': losses}
