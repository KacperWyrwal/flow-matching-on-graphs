"""Abstract ConfigurationSpace base class.

From config_fm/config_space.py.
"""

from abc import ABC, abstractmethod
import numpy as np


class ConfigurationSpace(ABC):
    """Abstract base class for configuration flow matching.

    A configuration is a labeled graph (G, ell) where G is a fixed
    position graph and ell: V -> vocab is a node labeling.

    Subclasses define the position graph, vocabulary, invariant,
    valid transitions, geodesic computation, and sampling.
    """

    @property
    @abstractmethod
    def n_positions(self) -> int:
        """Number of nodes in the position graph."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Size of the label vocabulary."""

    @property
    @abstractmethod
    def transition_order(self) -> int:
        """Number of nodes affected per transition.
        k=1 for DFM, k=2 for Johnson/Kawasaki."""

    @abstractmethod
    def position_graph_edges(self) -> np.ndarray:
        """Edge index of the position graph for GNN message passing.
        Shape: (2, n_edges)."""

    @abstractmethod
    def position_edge_features(self) -> np.ndarray | None:
        """Optional edge features for the position graph.
        Shape: (n_edges, d_edge) or None."""

    @abstractmethod
    def node_features(self, config: np.ndarray) -> np.ndarray:
        """Compute node features for the GNN given a configuration.
        Returns: (n_positions, d_node) feature array."""

    @abstractmethod
    def global_features(self, **kwargs) -> np.ndarray:
        """Compute global context features (e.g., [t, beta]).
        Returns: (d_global,) feature array."""

    def dynamic_edge_features(self, config: np.ndarray) -> np.ndarray | None:
        """Dynamic edge features depending on current configuration.
        Shape: (n_edges, d_dynamic_edge). Default: None.
        Override for edge-labeled problems (e.g., degree-preserving)."""
        return None

    @abstractmethod
    def transition_mask(self, config: np.ndarray) -> np.ndarray:
        """Mask over the model's output indicating valid transitions.
        Shape depends on transition_order:
          k=1: (n_positions,)
          k=2: (n_positions, n_positions)"""

    @abstractmethod
    def apply_transition(self, config: np.ndarray,
                         transition_idx) -> np.ndarray | None:
        """Apply a transition. Returns new config or None if invalid."""

    def enumerate_transitions(self, config: np.ndarray) -> list:
        """Enumerate valid transitions as structured descriptors.
        For k>=4, returns list of (a, b, c, d, rewiring) tuples.
        Default: not implemented (k=1, k=2 use mask-based API)."""
        raise NotImplementedError(
            "enumerate_transitions not implemented for this space")

    def apply_transition_by_descriptor(self, config: np.ndarray,
                                       descriptor) -> np.ndarray | None:
        """Apply a transition given its structured descriptor.
        Default: not implemented (k=1, k=2 use apply_transition)."""
        raise NotImplementedError(
            "apply_transition_by_descriptor not implemented for this space")

    def compute_target_rates_enumerated(self, config_0, config_T,
                                         config_t, t):
        """Returns (transitions, rates) for enumeration-based API.
        Default: not implemented (k=1, k=2 use compute_target_rates)."""
        raise NotImplementedError(
            "compute_target_rates_enumerated not implemented for this space")

    @abstractmethod
    def geodesic_distance(self, config_a: np.ndarray,
                          config_b: np.ndarray) -> int:
        """Geodesic distance on the configuration graph."""

    @abstractmethod
    def sample_intermediate(self, config_0: np.ndarray,
                            config_T: np.ndarray,
                            t: float, rng) -> tuple:
        """Sample intermediate configuration along geodesic.
        Returns: (config_t, n_completed, n_remaining)."""

    @abstractmethod
    def compute_target_rates(self, config_0: np.ndarray,
                             config_T: np.ndarray,
                             config_t: np.ndarray,
                             t: float) -> np.ndarray:
        """Compute conditional target rates at config_t.
        Does NOT include the 1/(1-t) factor."""

    @abstractmethod
    def sample_source(self, rng) -> np.ndarray:
        """Sample from the source distribution."""

    @abstractmethod
    def sample_target(self, rng, **kwargs) -> np.ndarray:
        """Sample from the target distribution."""
