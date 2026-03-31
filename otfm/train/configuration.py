"""Problem-independent training loop for configuration flow matching.

From config_fm/train.py.
"""

import numpy as np
import torch

from otfm.core.loss import rate_kl_loss


def _combine_edge_features(static_ef, dynamic_ef):
    """Combine static and dynamic edge features (numpy)."""
    if static_ef is not None and dynamic_ef is not None:
        return np.concatenate([static_ef, dynamic_ef], axis=-1)
    elif static_ef is not None:
        return static_ef
    elif dynamic_ef is not None:
        return dynamic_ef
    return None


def _rate_kl_loss_enumerated(pred_rates_list, target_rates_list):
    """Rate KL loss for enumerated transitions (k>=4).

    pred_rates_list: list of B tensors, each (n_swaps_b,)
    target_rates_list: list of B numpy arrays, each (n_swaps_b,)
    """
    eps = 1e-10
    total_loss = 0.0
    n_total = 0

    for pred, target_np in zip(pred_rates_list, target_rates_list):
        if len(pred) == 0:
            continue
        target = torch.tensor(target_np, dtype=torch.float32,
                              device=pred.device)
        active = target > eps
        if active.any():
            r_true = target[active]
            r_pred = pred[active].clamp(min=eps)
            total_loss = total_loss + (
                r_true * (r_true.clamp(min=eps).log() - r_pred.log())
                - r_true + r_pred).sum()
        # Penalize nonzero predictions where target is 0
        inactive = ~active
        if inactive.any():
            total_loss = total_loss + pred[inactive].sum()
        n_total += len(pred)

    return total_loss / max(n_total, 1)


def train_configuration_fm(model, config_space, target_sampler_kwargs,
                           n_epochs=2000, batch_size=256, lr=5e-4,
                           device='cpu', seed=42,
                           steps_per_epoch=None):
    """Train configuration flow matching model.

    Handles both mask-based (k=1, k=2) and enumeration-based (k>=4) APIs.
    Supports dynamic edge features.

    The config_space.sample_target(rng, **kwargs) may return either:
      - config (ndarray): target configuration
      - (config, context_kwargs): target + kwargs to pass to global_features
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    rng = np.random.default_rng(seed)

    edge_index = config_space.position_graph_edges()
    static_ef = config_space.position_edge_features()

    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)

    if steps_per_epoch is None:
        steps_per_epoch = max(1, 1000 // batch_size)

    use_enumeration = config_space.transition_order >= 4
    # Check if this space has dynamic edge features
    _test_config = config_space.sample_source(np.random.default_rng(0))
    has_dynamic_ef = config_space.dynamic_edge_features(_test_config) is not None
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            batch_nf, batch_gf = [], []

            if use_enumeration:
                batch_transitions = []
                batch_target_rates = []
            else:
                batch_tr, batch_mask = [], []

            batch_ef = []  # dynamic edge features per sample

            for _ in range(batch_size):
                config_0 = config_space.sample_source(rng)
                target_result = config_space.sample_target(
                    rng, **target_sampler_kwargs)

                if isinstance(target_result, tuple):
                    config_T, context_kwargs = target_result
                else:
                    config_T = target_result
                    context_kwargs = target_sampler_kwargs

                t = float(rng.uniform(0.0, 0.999))

                config_t, _, _ = config_space.sample_intermediate(
                    config_0, config_T, t, rng)

                node_feat = config_space.node_features(config_t)
                global_feat = config_space.global_features(
                    t=t, **context_kwargs)

                # Dynamic edge features
                dynamic_ef = config_space.dynamic_edge_features(config_t)
                ef = _combine_edge_features(static_ef, dynamic_ef)
                batch_ef.append(ef)

                batch_nf.append(node_feat)
                batch_gf.append(global_feat)

                if use_enumeration:
                    transitions, target_rates = \
                        config_space.compute_target_rates_enumerated(
                            config_0, config_T, config_t, t)
                    batch_transitions.append(transitions)
                    batch_target_rates.append(target_rates)
                else:
                    target_rates = config_space.compute_target_rates(
                        config_0, config_T, config_t, t)
                    mask = config_space.transition_mask(config_t)
                    batch_tr.append(target_rates)
                    batch_mask.append(mask)

            nf_t = torch.tensor(np.array(batch_nf),
                                dtype=torch.float32, device=device)
            gf_t = torch.tensor(np.array(batch_gf),
                                dtype=torch.float32, device=device)

            # Edge features: (B, E, d) if dynamic, (E, d) if static only
            if batch_ef[0] is not None:
                if has_dynamic_ef:
                    # Per-sample: batch as (B, E, d)
                    ef_t = torch.tensor(np.array(batch_ef),
                                        dtype=torch.float32, device=device)
                else:
                    # Static: share across batch (E, d)
                    ef_t = torch.tensor(batch_ef[0],
                                        dtype=torch.float32, device=device)
            else:
                ef_t = None

            if use_enumeration:
                pred_rates_list = model.score_transitions(
                    nf_t, edge_index_t, ef_t, gf_t, batch_transitions)
                loss = _rate_kl_loss_enumerated(
                    pred_rates_list, batch_target_rates)
            else:
                tr_t = torch.tensor(np.array(batch_tr),
                                    dtype=torch.float32, device=device)
                mask_t = torch.tensor(np.array(batch_mask),
                                      dtype=torch.float32, device=device)
                pred_rates = model(nf_t, edge_index_t, ef_t, gf_t, mask_t)
                loss = rate_kl_loss(pred_rates, tr_t, mask_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        losses.append(avg_loss)

        print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f}",
              flush=True)

    return {'losses': losses}
