from typing import Iterable
import numpy as np

from atomic_datasets import datatypes


def compute_minimum_distance_to_first_node(solid: np.ndarray) -> float:
    """Computes the distance between the first node and its closest neighbor."""
    return np.min(np.linalg.norm(solid[0] - solid[1:], axis=-1))


def to_graph(solid: np.ndarray) -> datatypes.Graph:
    """Converts a solid to a datatypes.Graph object."""
    return datatypes.Graph(
        nodes=dict(
            positions=np.asarray(solid),
            species=np.zeros(len(solid), dtype=int),
        ),
        edges=None,
        receivers=None,
        senders=None,
        globals=None,
        n_node=np.asarray([len(solid)]),
        n_edge=None,
    )


class PlatonicSolidsDataset(datatypes.MolecularDataset):
    """Dataset of platonic solids."""

    def __init__(
        self,
    ):
        super().__init__()
        self.all_graphs = list(load_platonic_solids())

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1])

    def __iter__(self) -> Iterable[datatypes.Graph]:
        for graph in self.all_graphs:
            yield graph
    
    def __len__(self) -> int:
        return len(self.all_graphs)
    
    def __getitem__(self, idx: int) -> datatypes.Graph:
        return self.all_graphs[idx]



def load_platonic_solids():
    """Obtained from https://en.wikipedia.org/wiki/Platonic_solid."""
    PHI = (1 + np.sqrt(5)) / 2
    solids = [
        [
            (1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1)
        ],  # tetrahedron
        [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ],  # octahedron
        [
            (1, 1, 1),
            (-1, 1, 1),
            (1, -1, 1),
            (1, 1, -1),
            (-1, -1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, -1),
        ],  # cube
        [
            (0, 1, PHI),
            (0, -1, PHI),
            (0, 1, -PHI),
            (0, -1, -PHI),
            (1, PHI, 0),
            (-1, PHI, 0),
            (1, -PHI, 0),
            (-1, -PHI, 0),
            (PHI, 0, 1),
            (PHI, 0, -1),
            (-PHI, 0, 1),
            (-PHI, 0, -1),
        ],  # icosahedron
        [
            (1, 1, 1),
            (-1, 1, 1),
            (1, -1, 1),
            (1, 1, -1),
            (-1, -1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, -1),
            (0, 1 / PHI, PHI),
            (0, -1 / PHI, PHI),
            (0, 1 / PHI, -PHI),
            (0, -1 / PHI, -PHI),
            (1 / PHI, PHI, 0),
            (-1 / PHI, PHI, 0),
            (1 / PHI, -PHI, 0),
            (-1 / PHI, -PHI, 0),
            (PHI, 0, 1 / PHI),
            (PHI, 0, -1 / PHI),
            (-PHI, 0, 1 / PHI),
            (-PHI, 0, -1 / PHI),
        ],  # dodacahedron
    ]
    for solid in solids:
        solid = to_graph(np.asarray(solid))
        
        # Normalize the solid edges, so that the smallest inter-node distance is 1.
        scale_factor = compute_minimum_distance_to_first_node(solid["nodes"]["positions"])
        solid["nodes"]["positions"] /= scale_factor

        yield solid
