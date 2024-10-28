from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import jraph

from atomic_datasets import InMemoryDataset
from atomic_datasets.datasets.platonic_solids import _solid_to_structure


class TetrisDataset(InMemoryDataset):
    """Dataset of 3D Tetris shapes."""

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

    def structures(self) -> List[jraph.GraphsTuple]:
        """Returns the structures for the Platonic solids."""
        # Taken from https://docs.e3nn.org/en/stable/examples/tetris_gate.html.
        solids = [
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],   # chiral_shape_1
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],   # square
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],   # line
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],   # corner
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],   # L
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],   # T
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],   # zigzag
        ]

        # Convert to Structures.
        structures = [_solid_to_structure(solid) for solid in solids]
        return structures

    def split_indices(self) -> Dict[str, Set[int]]:
        """Returns the split indices for the Platonic solids."""
        return {
            "train": self.train_solids,
            "val": self.val_solids,
            "test": self.test_solids,
        }
