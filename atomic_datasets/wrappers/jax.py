"""Jraph wrapper for atomic datasets.

Usage:
    from atomic_datasets import Tetris
    from atomic_datasets.wrappers.jax import JraphDataset

    dataset = JraphDataset(Tetris())
    graph = dataset[0]  # jraph.GraphsTuple

    # Batch multiple graphs:
    import jraph
    batch = jraph.batch_np([dataset[i] for i in range(4)])

    # Pad for JIT compatibility:
    padded = jraph.pad_with_graphs(batch, n_node=20, n_edge=0, n_graph=5)
"""

import numpy as np

try:
    import jax.numpy as jnp
    import jraph
except ImportError:
    raise ImportError(
        "JAX and Jraph are required for this wrapper. "
        "Install with: pip install jax jraph"
    )

from atomic_datasets.datatypes import MolecularDataset


def graph_to_graphs_tuple(graph: dict) -> jraph.GraphsTuple:
    """Convert a single atomic_datasets Graph dict to a jraph.GraphsTuple.

    Converts all node attributes, edge info, and globals present in the
    source graph.
    """
    nodes = graph["nodes"]
    positions = np.array(nodes["positions"], dtype=np.float32)
    species = np.array(nodes["species"], dtype=np.int32)
    n_node = positions.shape[0]

    node_features = {"positions": positions, "species": species}

    # Carry over any extra node attributes
    for key, val in nodes.items():
        if key in ("positions", "species"):
            continue
        node_features[key] = np.asarray(val)

    # Edges
    if graph.get("senders") is not None and graph.get("receivers") is not None:
        senders = np.array(graph["senders"], dtype=np.int32)
        receivers = np.array(graph["receivers"], dtype=np.int32)
        n_edge = len(senders)
        edges = None
        if graph.get("edges") is not None:
            edges = np.array(graph["edges"], dtype=np.float32)
    else:
        senders = np.array([], dtype=np.int32)
        receivers = np.array([], dtype=np.int32)
        n_edge = 0
        edges = None

    # Globals
    globals_ = {}
    if graph.get("globals") is not None:
        val = np.asarray(graph["globals"])
        if val.ndim == 0:
            val = val[None]  # Make scalar globals 1D for Jraph
        globals_["globals"] = val

    # Add properties as globals if present
    if graph.get("properties") is not None:
        globals_["properties"] = {}
        for key, val in graph["properties"].items():
            val = np.asarray(val)
            if val.ndim == 0:
                val = val[None]  # Make scalar properties 1D for Jraph
            globals_["properties"][key] = val

    return jraph.GraphsTuple(
        nodes=node_features,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals_,
        n_node=np.array([n_node]),
        n_edge=np.array([n_edge]),
    )


class JraphDataset:
    """Wraps any ``MolecularDataset`` for use with Jraph.

    Args:
        dataset: Any ``MolecularDataset`` instance.
    """

    def __init__(self, dataset: MolecularDataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> jraph.GraphsTuple:
        return graph_to_graphs_tuple(self._dataset[idx])
