from typing import List, Set, Dict, Sequence, Optional

import numpy as np
import jraph

from atomic_datasets import datatypes


def _compute_first_node_min_distance(solid: np.ndarray) -> float:
    """Computes the distance between the first node and its closest neighbor."""
    return np.min(np.linalg.norm(solid[0] - solid[1:], axis=-1))


def _solid_to_structure(solid: np.ndarray) -> jraph.GraphsTuple:
    """Converts a solid to a jraph.GraphsTuple object."""
    return jraph.GraphsTuple(
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


class PlatonicSolidsDataset(datatypes.InMemoryMolecularDataset):
    """Dataset of platonic solids."""

    def __init__(
        self,
        train_solids: Optional[Sequence[int]],
        val_solids: Optional[Sequence[int]],
        test_solids: Optional[Sequence[int]],
    ):
        super().__init__()

        all_indices = range(5)
        if train_solids is None:
            train_solids = all_indices
        if val_solids is None:
            val_solids = all_indices
        if test_solids is None:
            test_solids = all_indices

        self.train_solids = train_solids
        self.val_solids = val_solids
        self.test_solids = test_solids

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1])

    def __iter__(self) -> List[GraphData]:
        """Returns the structures for the Platonic solids."""
        # Taken from Wikipedia.
        # https://en.wikipedia.org/wiki/Platonic_solid
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

        # Normalize the solids, so that the smallest inter-node distance is 1.
        solids_as_arrays = [np.asarray(solid) for solid in solids]
        scale_factors = [
            1 / np.min(_compute_first_node_min_distance(solid))
            for solid in solids_as_arrays
        ]
        solids = [
            solid * factor for solid, factor in zip(solids_as_arrays, scale_factors)
        ]

        # Convert to Structures.
        structures = [_solid_to_structure(solid) for solid in solids]
        return iter(structures)

    def split_indices(self) -> Dict[str, Set[int]]:
        """Returns the split indices for the Platonic solids."""
        return {
            "train": self.train_solids,
            "val": self.val_solids,
            "test": self.test_solids,
        }
