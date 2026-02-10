"""Tests for the Jraph wrapper."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jraph = pytest.importorskip("jraph")
jnp = jax.numpy

from atomic_datasets import Tetris, PlatonicSolids
from atomic_datasets.wrappers.jax import JraphDataset, graph_to_graphs_tuple


@pytest.fixture
def tetris():
    return Tetris()


@pytest.fixture
def platonic():
    return PlatonicSolids()


class TestGraphToGraphsTuple:
    """Test the low-level conversion function."""

    def test_basic_conversion(self, tetris):
        gt = graph_to_graphs_tuple(tetris[0])

        assert isinstance(gt, jraph.GraphsTuple)
        assert gt.nodes["positions"].shape == (4, 3)
        assert gt.nodes["species"].shape == (4,)
        assert gt.n_node.item() == 4

    def test_no_edges_when_source_has_none(self, tetris):
        gt = graph_to_graphs_tuple(tetris[0])
        assert gt.n_edge.item() == 0
        assert len(gt.senders) == 0
        assert len(gt.receivers) == 0
        assert gt.edges is None

    def test_carries_over_atom_types(self, tetris):
        gt = graph_to_graphs_tuple(tetris[0])
        assert "atom_types" in gt.nodes
        assert len(gt.nodes["atom_types"]) == 4
        assert all(t == "H" for t in gt.nodes["atom_types"])

    def test_carries_over_edges_when_present(self):
        graph = {
            "nodes": {
                "positions": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
                "species": np.array([0, 0]),
            },
            "senders": np.array([0, 1]),
            "receivers": np.array([1, 0]),
            "edges": np.array([[1.0], [1.0]]),
            "n_node": np.array([2]),
            "n_edge": np.array([2]),
            "globals": None,
        }
        gt = graph_to_graphs_tuple(graph)
        assert gt.n_edge.item() == 2
        assert gt.edges.shape == (2, 1)

    def test_carries_over_globals(self):
        graph = {
            "nodes": {
                "positions": np.array([[0, 0, 0]], dtype=np.float32),
                "species": np.array([0]),
            },
            "globals": np.array([1.0, 2.0]),
        }
        gt = graph_to_graphs_tuple(graph)
        np.testing.assert_allclose(gt.globals, jnp.array([1.0, 2.0]))


class TestJraphDataset:
    """Test the Dataset wrapper."""

    def test_len(self, tetris):
        ds = JraphDataset(tetris)
        assert len(ds) == 8

    def test_getitem(self, tetris):
        ds = JraphDataset(tetris)
        gt = ds[0]
        assert isinstance(gt, jraph.GraphsTuple)
        assert gt.nodes["positions"].shape == (4, 3)

    def test_all_items(self, tetris):
        ds = JraphDataset(tetris)
        for i in range(len(ds)):
            gt = ds[i]
            assert gt.nodes["positions"].shape[0] == 4
            assert gt.nodes["species"].shape[0] == 4

    def test_platonic_solids_sizes(self, platonic):
        ds = JraphDataset(platonic)
        expected_sizes = [4, 6, 8, 12, 20]
        for i, expected in enumerate(expected_sizes):
            gt = ds[i]
            assert gt.n_node.item() == expected

    def test_iteration(self, tetris):
        ds = JraphDataset(tetris)
        items = list(ds)
        assert len(items) == 8

    def test_batch(self, tetris):
        ds = JraphDataset(tetris)
        batch = jraph.batch([ds[i] for i in range(4)])
        # 4 tetris pieces × 4 atoms = 16 atoms total
        assert batch.nodes["positions"].shape[0] == 16
        assert batch.nodes["species"].shape[0] == 16
        assert batch.n_node.shape[0] == 4