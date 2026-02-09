"""PyTorch Geometric wrapper for atomic datasets.

Usage:
    from atomic_datasets import Tetris
    from atomic_datasets.wrappers import PyTorchGeometricDataset

    dataset = PyTorchGeometricDataset(Tetris())
    data = dataset[0]  # torch_geometric.data.Data

    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        print(batch.pos.shape)      # (total_atoms_in_batch, 3)
        print(batch.species.shape)  # (total_atoms_in_batch,)
        print(batch.batch.shape)    # (total_atoms_in_batch,) — maps atom -> molecule
"""

import numpy as np

try:
    import torch
    from torch_geometric.data import Data, Dataset
except ImportError:
    raise ImportError(
        "PyTorch Geometric is required for this wrapper. "
        "Install with: pip install atomic-datasets[torch]"
    )

from atomic_datasets.datatypes import MolecularDataset


def graph_to_data(graph: dict) -> Data:
    """Convert a single atomic_datasets Graph dict to a PyG Data object.

    Converts all numeric node attributes, graph properties, and edge info
    present in the source graph. String values are stored as Python lists.
    """
    nodes = graph["nodes"]
    positions = np.asarray(nodes["positions"], dtype=np.float32)
    species = np.asarray(nodes["species"], dtype=np.int64)

    data = Data(
        pos=torch.tensor(positions, dtype=torch.float32),
        species=torch.tensor(species, dtype=torch.long),
        num_nodes=len(positions),
    )

    # Carry over any extra node attributes
    for key, val in nodes.items():
        if key in ("positions", "species"):
            continue
        val = np.asarray(val)
        if val.dtype.kind in ("U", "S", "O"):
            setattr(data, key, val.tolist())
        else:
            t = torch.tensor(np.array(val))
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            setattr(data, key, t)

    # Carry over edges if present
    if graph.get("senders") is not None and graph.get("receivers") is not None:
        senders = np.asarray(graph["senders"], dtype=np.int64)
        receivers = np.asarray(graph["receivers"], dtype=np.int64)
        data.edge_index = torch.tensor(
            np.stack([senders, receivers], axis=0), dtype=torch.long
        )

    if graph.get("edges") is not None:
        edges = np.asarray(graph["edges"], dtype=np.float32)
        data.edge_attr = torch.tensor(edges, dtype=torch.float32)

    # Carry over graph-level properties if present
    if graph.get("properties") is not None:
        for key, val in graph["properties"].items():
            if isinstance(val, str):
                setattr(data, key, val)
            elif isinstance(val, (int, float, np.integer, np.floating)):
                setattr(data, key, torch.tensor([val], dtype=torch.float32))

    if graph.get("globals") is not None:
        data.globals = torch.tensor(
            np.asarray(graph["globals"], dtype=np.float32), dtype=torch.float32
        )

    return data


class PyTorchGeometricDataset(Dataset):
    """Wraps any ``MolecularDataset`` as a PyTorch Geometric ``Dataset``.

    Args:
        dataset: Any ``MolecularDataset`` instance.
    """

    def __init__(self, dataset: MolecularDataset):
        self._dataset = dataset

    def len(self) -> int:
        return len(self._dataset)

    def get(self, idx: int) -> Data:
        return graph_to_data(self._dataset[idx])