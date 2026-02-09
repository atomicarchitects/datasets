"""Tests for the PyTorch Geometric wrapper."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")

from atomic_datasets import Tetris, PlatonicSolids
from atomic_datasets.wrappers.pyg import PyTorchGeometricDataset, graph_to_data


@pytest.fixture
def tetris():
    return Tetris()


@pytest.fixture
def platonic():
    return PlatonicSolids()


class TestGraphToData:
    """Test the low-level conversion function."""

    def test_basic_conversion(self, tetris):
        data = graph_to_data(tetris[0])

        assert isinstance(data, pyg.data.Data)
        assert data.pos.shape == (4, 3)
        assert data.species.shape == (4,)
        assert data.pos.dtype == torch.float32
        assert data.species.dtype == torch.long

    def test_no_edges_when_source_has_none(self, tetris):
        data = graph_to_data(tetris[0])
        assert data.edge_index is None

    def test_carries_over_atom_types(self, tetris):
        data = graph_to_data(tetris[0])
        assert hasattr(data, "atom_types")
        assert len(data.atom_types) == 4
        assert all(t == "H" for t in data.atom_types)

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
        data = graph_to_data(graph)
        assert data.edge_index.shape == (2, 2)
        assert data.edge_attr.shape == (2, 1)

    def test_carries_over_properties(self):
        graph = {
            "nodes": {
                "positions": np.array([[0, 0, 0]], dtype=np.float32),
                "species": np.array([0]),
            },
            "properties": {"smiles": "C", "energy": -42.0},
            "n_node": np.array([1]),
        }
        data = graph_to_data(graph)
        assert data.smiles == "C"
        assert data.energy.item() == pytest.approx(-42.0)

    def test_carries_over_globals(self):
        graph = {
            "nodes": {
                "positions": np.array([[0, 0, 0]], dtype=np.float32),
                "species": np.array([0]),
            },
            "globals": np.array([1.0, 2.0]),
        }
        data = graph_to_data(graph)
        torch.testing.assert_close(data.globals, torch.tensor([1.0, 2.0]))


class TestPyTorchGeometricDataset:
    """Test the Dataset wrapper."""

    def test_len(self, tetris):
        ds = PyTorchGeometricDataset(tetris)
        assert len(ds) == 8

    def test_getitem(self, tetris):
        ds = PyTorchGeometricDataset(tetris)
        data = ds[0]
        assert isinstance(data, pyg.data.Data)
        assert data.pos.shape == (4, 3)

    def test_all_items(self, tetris):
        ds = PyTorchGeometricDataset(tetris)
        for i in range(len(ds)):
            data = ds[i]
            assert data.pos.shape[0] == 4
            assert data.species.shape[0] == 4

    def test_platonic_solids_sizes(self, platonic):
        ds = PyTorchGeometricDataset(platonic)
        expected_sizes = [4, 6, 8, 12, 20]
        for i, expected in enumerate(expected_sizes):
            data = ds[i]
            assert data.num_nodes == expected

    def test_dataloader_batching(self, tetris):
        from torch_geometric.loader import DataLoader

        ds = PyTorchGeometricDataset(tetris)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        # 4 tetris pieces × 4 atoms = 16 atoms total
        assert batch.pos.shape == (16, 3)
        assert batch.species.shape == (16,)
        assert batch.batch.shape == (16,)
        assert set(batch.batch.tolist()) == {0, 1, 2, 3}

    def test_dataloader_full_epoch(self, tetris):
        from torch_geometric.loader import DataLoader

        ds = PyTorchGeometricDataset(tetris)
        loader = DataLoader(ds, batch_size=3, shuffle=False)
        batches = list(loader)
        # 8 items, batch_size=3 -> 3 batches (3, 3, 2)
        assert len(batches) == 3
        total_graphs = sum(b.num_graphs for b in batches)
        assert total_graphs == 8