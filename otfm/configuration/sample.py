"""Problem-independent inference / generation for configuration FM.

From config_fm/sample.py.
"""

import numpy as np
import torch


def _combine_edge_features(static_ef, dynamic_ef):
    """Combine static and dynamic edge features (numpy)."""
    if static_ef is not None and dynamic_ef is not None:
        return np.concatenate([static_ef, dynamic_ef], axis=-1)
    elif static_ef is not None:
        return static_ef
    elif dynamic_ef is not None:
        return dynamic_ef
    return None


def generate_samples(model, config_space, n_samples, n_steps=100,
                     device='cpu', seed=0, batch_size=512, **kwargs):
    """Generate samples via learned configuration flow matching.

    Handles both mask-based (k=1, k=2) and enumeration-based (k>=4) APIs.
    Supports dynamic edge features.

    Args:
        model: trained ConfigurationRatePredictor
        config_space: ConfigurationSpace instance
        n_samples: number of samples to generate
        n_steps: number of integration steps
        device: torch device
        batch_size: samples per batch for model forward pass
        **kwargs: passed to config_space.global_features()
    Returns: (n_samples, ...) array of configurations
    """
    model.eval()
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_steps

    edge_index = config_space.position_graph_edges()
    static_ef = config_space.position_edge_features()
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)

    use_enumeration = config_space.transition_order >= 4
    # Check if this space has dynamic edge features
    _test = config_space.sample_source(np.random.default_rng(0))
    has_dynamic_ef = config_space.dynamic_edge_features(_test) is not None
    all_samples = []

    for batch_start in range(0, n_samples, batch_size):
        B = min(batch_size, n_samples - batch_start)
        configs = np.array([config_space.sample_source(rng)
                            for _ in range(B)])

        with torch.no_grad():
            for step in range(n_steps):
                t = step * dt

                node_feats = torch.tensor(
                    np.array([config_space.node_features(c)
                              for c in configs]),
                    dtype=torch.float32, device=device)
                global_feats = torch.tensor(
                    np.array([config_space.global_features(t=t, **kwargs)
                              for _ in range(B)]),
                    dtype=torch.float32, device=device)

                if use_enumeration:
                    # k>=4: per-sample forward (enumeration-based)
                    for b in range(B):
                        transitions = config_space.enumerate_transitions(
                            configs[b])
                        if not transitions:
                            continue

                        nf_b = node_feats[b:b+1]
                        gf_b = global_feats[b:b+1]

                        dyn_ef = config_space.dynamic_edge_features(configs[b])
                        ef_b = _combine_edge_features(static_ef, dyn_ef)
                        ef_bt = (torch.tensor(ef_b, dtype=torch.float32,
                                              device=device)
                                 if ef_b is not None else None)

                        rates_list = model.score_transitions(
                            nf_b, edge_index_t, ef_bt, gf_b, [transitions])
                        rates = rates_list[0] / (1.0 - t + 1e-10)
                        rates_np = rates.cpu().numpy()

                        total_rate = rates_np.sum()
                        if total_rate > 0:
                            n_events = rng.poisson(total_rate * dt)
                            if n_events > 0:
                                probs = rates_np / total_rate
                                swap_idx = rng.choice(len(transitions),
                                                      p=probs)
                                new_config = \
                                    config_space.apply_transition_by_descriptor(
                                        configs[b], transitions[swap_idx])
                                if new_config is not None:
                                    configs[b] = new_config
                else:
                    # k=1, k=2: mask-based batched forward
                    # Compute edge features
                    if has_dynamic_ef:
                        ef_list = [
                            _combine_edge_features(
                                static_ef,
                                config_space.dynamic_edge_features(c))
                            for c in configs]
                        ef_t = torch.tensor(np.array(ef_list),
                                            dtype=torch.float32, device=device)
                    elif static_ef is not None:
                        ef_t = torch.tensor(static_ef, dtype=torch.float32,
                                            device=device)
                    else:
                        ef_t = None

                    masks = torch.tensor(
                        np.array([config_space.transition_mask(c)
                                  for c in configs]),
                        dtype=torch.float32, device=device)

                    rates = model(node_feats, edge_index_t, ef_t,
                                  global_feats, masks)
                    rates = rates / (1.0 - t + 1e-10)
                    rates_np = rates.cpu().numpy()

                    for b in range(B):
                        rate_flat = rates_np[b].flatten()
                        total_rate = rate_flat.sum()
                        if total_rate <= 0:
                            continue
                        n_events = rng.poisson(total_rate * dt)
                        if n_events > 0:
                            probs = rate_flat / total_rate
                            idx = rng.choice(len(probs), p=probs)
                            new_config = config_space.apply_transition(
                                configs[b], idx)
                            if new_config is not None:
                                configs[b] = new_config

        all_samples.append(configs)

    return np.concatenate(all_samples, axis=0)
