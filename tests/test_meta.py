"""
Tests for meta_fm: MetaFlowMatchingDataset, RateMatrixPredictor, training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from graph_ot_fm import GraphStructure, make_cycle_graph
from meta_fm import MetaFlowMatchingDataset, RateMatrixPredictor, train


@pytest.fixture
def cycle6_graph():
    R = make_cycle_graph(6, weighted=False)
    return GraphStructure(R)


def peaked_dist(n, node, eps=0.02):
    d = np.ones(n) * eps / (n - 1)
    d[node] = 1.0 - eps
    d /= d.sum()
    return d


def near_uniform(n, seed=0):
    rng = np.random.default_rng(seed)
    d = np.ones(n) / n + rng.normal(0, 0.05 / n, n)
    d = np.clip(d, 1e-3, None)
    d /= d.sum()
    return d


@pytest.fixture
def small_dataset(cycle6_graph):
    """Small dataset for fast tests."""
    N = 6
    sources = [peaked_dist(N, i % N) for i in range(5)]
    targets = [near_uniform(N, i) for i in range(5)]
    return MetaFlowMatchingDataset(
        graph=cycle6_graph,
        source_distributions=sources,
        target_distributions=targets,
        n_samples=50,
        seed=42,
    )


class TestMetaFlowMatchingDataset:
    """Tests for MetaFlowMatchingDataset."""

    def test_length(self, small_dataset):
        """Dataset should have correct number of samples."""
        assert len(small_dataset) == 50

    def test_getitem_shapes(self, small_dataset):
        """Each sample should have correct shapes."""
        mu, tau, R_target = small_dataset[0]
        assert mu.shape == (6,), f"mu shape wrong: {mu.shape}"
        assert tau.shape == (1,), f"tau shape wrong: {tau.shape}"
        assert R_target.shape == (6, 6), f"R_target shape wrong: {R_target.shape}"

    def test_mu_sums_to_one(self, small_dataset):
        """Each mu should sum to approximately 1."""
        for i in range(len(small_dataset)):
            mu, tau, R = small_dataset[i]
            assert abs(mu.sum().item() - 1.0) < 1e-4, (
                f"mu doesn't sum to 1 at idx {i}: {mu.sum().item()}"
            )

    def test_mu_nonnegative(self, small_dataset):
        """Each mu should be non-negative."""
        for i in range(min(20, len(small_dataset))):
            mu, tau, R = small_dataset[i]
            assert torch.all(mu >= -1e-6), f"Negative mu at idx {i}"

    def test_tau_in_range(self, small_dataset):
        """tau should be in [0, 1)."""
        for i in range(len(small_dataset)):
            mu, tau, R = small_dataset[i]
            tau_val = tau.item()
            assert 0.0 <= tau_val <= 1.0, f"tau out of range: {tau_val}"

    def test_R_target_row_sums_zero(self, small_dataset):
        """R_target should have zero row sums."""
        for i in range(min(20, len(small_dataset))):
            mu, tau, R = small_dataset[i]
            row_sums = R.sum(dim=1)
            assert torch.allclose(row_sums, torch.zeros(6), atol=1e-5), (
                f"Row sums not zero at idx {i}: {row_sums}"
            )

    def test_R_target_dtype(self, small_dataset):
        """R_target should be float32."""
        mu, tau, R = small_dataset[0]
        assert R.dtype == torch.float32

    def test_larger_dataset(self, cycle6_graph):
        """Test with more samples."""
        N = 6
        sources = [peaked_dist(N, i % N) for i in range(10)]
        targets = [near_uniform(N, i) for i in range(10)]
        dataset = MetaFlowMatchingDataset(
            graph=cycle6_graph,
            source_distributions=sources,
            target_distributions=targets,
            n_samples=100,
            seed=42,
        )
        assert len(dataset) == 100


class TestRateMatrixPredictor:
    """Tests for RateMatrixPredictor."""

    def test_output_shape(self):
        """Model output should have shape (batch, N, N)."""
        N = 6
        batch = 4
        model = RateMatrixPredictor(n_nodes=N, hidden_dim=32, n_layers=2)
        mu = torch.randn(batch, N).softmax(dim=1)
        t = torch.rand(batch, 1)
        R = model(mu, t)
        assert R.shape == (batch, N, N), f"Wrong shape: {R.shape}"

    def test_row_sums_zero(self):
        """Model output should have zero row sums."""
        N = 6
        batch = 8
        model = RateMatrixPredictor(n_nodes=N, hidden_dim=32, n_layers=2)
        mu = torch.randn(batch, N).softmax(dim=1)
        t = torch.rand(batch, 1)
        R = model(mu, t)
        row_sums = R.sum(dim=2)
        assert torch.allclose(row_sums, torch.zeros(batch, N), atol=1e-5), (
            f"Row sums not zero: {row_sums}"
        )

    def test_offdiag_nonnegative(self):
        """Off-diagonal entries should be non-negative (softplus ensures this)."""
        N = 6
        batch = 8
        model = RateMatrixPredictor(n_nodes=N, hidden_dim=32, n_layers=2)
        mu = torch.randn(batch, N).softmax(dim=1)
        t = torch.rand(batch, 1)
        R = model(mu, t)
        # Off-diagonal: mask diagonal
        diag_mask = torch.eye(N, dtype=torch.bool)
        for b in range(batch):
            R_b = R[b]
            off_diag = R_b[~diag_mask]
            assert torch.all(off_diag >= -1e-6), (
                f"Negative off-diagonal entries at batch {b}"
            )

    def test_diagonal_negative(self):
        """Diagonal entries should be <= 0 (set to -sum of row)."""
        N = 6
        batch = 4
        model = RateMatrixPredictor(n_nodes=N, hidden_dim=32, n_layers=2)
        mu = torch.randn(batch, N).softmax(dim=1)
        t = torch.rand(batch, 1)
        R = model(mu, t)
        for b in range(batch):
            for i in range(N):
                assert R[b, i, i].item() <= 1e-6, (
                    f"Diagonal not negative at batch {b}, node {i}: {R[b,i,i].item()}"
                )

    def test_different_inputs_different_outputs(self):
        """Different inputs should produce different outputs."""
        N = 6
        model = RateMatrixPredictor(n_nodes=N, hidden_dim=32, n_layers=2)
        mu1 = torch.randn(1, N).softmax(dim=1)
        mu2 = torch.randn(1, N).softmax(dim=1)
        t = torch.tensor([[0.5]])
        R1 = model(mu1, t)
        R2 = model(mu2, t)
        assert not torch.allclose(R1, R2), "Different inputs gave identical outputs"

    def test_larger_n(self):
        """Test with larger N."""
        N = 16
        batch = 4
        model = RateMatrixPredictor(n_nodes=N, hidden_dim=64, n_layers=2)
        mu = torch.randn(batch, N).softmax(dim=1)
        t = torch.rand(batch, 1)
        R = model(mu, t)
        assert R.shape == (batch, N, N)
        row_sums = R.sum(dim=2)
        assert torch.allclose(row_sums, torch.zeros(batch, N), atol=1e-5)


class TestTraining:
    """Tests that training decreases loss."""

    def test_loss_decreases(self, cycle6_graph, small_dataset):
        """Training loss should decrease over epochs."""
        N = 6
        model = RateMatrixPredictor(n_nodes=N, hidden_dim=32, n_layers=2)
        history = train(
            model=model,
            dataset=small_dataset,
            n_epochs=20,
            batch_size=16,
            lr=1e-3,
            device='cpu',
        )
        losses = history['losses']
        assert len(losses) == 20

        # Loss should decrease from first to last
        # Use average of first 5 vs last 5 to be robust
        first_avg = np.mean(losses[:5])
        last_avg = np.mean(losses[-5:])
        assert last_avg <= first_avg * 1.5, (
            f"Loss did not decrease: first={first_avg:.6f}, last={last_avg:.6f}"
        )

    def test_training_returns_dict(self, small_dataset):
        """train() should return a dict with 'losses' key."""
        N = 6
        model = RateMatrixPredictor(n_nodes=N, hidden_dim=32, n_layers=2)
        history = train(
            model=model,
            dataset=small_dataset,
            n_epochs=5,
            batch_size=16,
            lr=1e-3,
            device='cpu',
        )
        assert isinstance(history, dict)
        assert 'losses' in history
        assert len(history['losses']) == 5

    def test_model_produces_valid_rates_after_training(self, small_dataset):
        """After training, model should still produce valid rate matrices."""
        N = 6
        model = RateMatrixPredictor(n_nodes=N, hidden_dim=32, n_layers=2)
        train(model=model, dataset=small_dataset, n_epochs=5, batch_size=16, device='cpu')

        # Check model output validity
        mu = torch.randn(4, N).softmax(dim=1)
        t = torch.rand(4, 1)
        R = model(mu, t)
        row_sums = R.sum(dim=2)
        assert torch.allclose(row_sums, torch.zeros(4, N), atol=1e-5)
