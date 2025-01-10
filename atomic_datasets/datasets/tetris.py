from typing import Iterable

import numpy as np

from atomic_datasets import datatypes
from atomic_datasets.datasets.platonic_solids import to_graph


class TetrisDataset(datatypes.MolecularDataset):
    """Dataset of 3D Tetris shapes."""

    def __init__(
        self,
    ):
        super().__init__()

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1])

    def __iter__(self) -> Iterable[datatypes.Graph]:
        """Returns the graphs for the Tetris pieces."""
        while True:
            yield from load_tetris()


def load_tetris() -> Iterable[datatypes.Graph]:
    """3D tetris pieces as taken from https://docs.e3nn.org/en/stable/examples/tetris_gate.html."""
    tetris_pieces = [
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],   # chiral_shape_1
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],   # square
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],   # line
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],   # corner
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],   # L
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],   # T
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],   # zigzag
    ]
    for piece in tetris_pieces:
        yield to_graph(piece)