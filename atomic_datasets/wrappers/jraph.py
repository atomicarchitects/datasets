"""Jraph wrapper for atomic datasets.

Usage:
    from atomic_datasets import Tetris
    from atomic_datasets.wrappers.jraph import JraphDataset

    dataset = JraphDataset(Tetris())
    graph = dataset[0]  # jraph.GraphsTuple

    # Batch multiple graphs:
    import jraph
    batch = jraph.batch([dataset[i] for i in range(4)])

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
    positions = jnp.array(nodes["positions"], dtype=jnp.float32)
    species = jnp.array(nodes["species"], dtype=jnp.int32)
    n_node = positions.shape[0]

    node_features = {"positions": positions, "species": species}

    # Carry over any extra node attributes
    for key, val in nodes.items():
        if key in ("positions", "species"):
            continue
        val = np.asarray(val)
        if val.dtype.kind in ("U", "S", "O"):
            node_features[key] = val.tolist()
        else:
            node_features[key] = jnp.array(val)

    # Edges
    if graph.get("senders") is not None and graph.get("receivers") is not None:
        senders = jnp.array(graph["senders"], dtype=jnp.int32)
        receivers = jnp.array(graph["receivers"], dtype=jnp.int32)
        n_edge = len(senders)
        edges = None
        if graph.get("edges") is not None:
            edges = jnp.array(graph["edges"], dtype=jnp.float32)
    else:
        senders = jnp.array([], dtype=jnp.int32)
        receivers = jnp.array([], dtype=jnp.int32)
        n_edge = 0
        edges = None

    # Globals
    globals_ = None
    if graph.get("globals") is not None:
        globals_ = jnp.array(graph["globals"], dtype=jnp.float32)

    return jraph.GraphsTuple(
        nodes=node_features,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals_,
        n_node=jnp.array([n_node]),
        n_edge=jnp.array([n_edge]),
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

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]