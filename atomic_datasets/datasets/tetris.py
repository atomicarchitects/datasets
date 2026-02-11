from typing import Iterable

import numpy as np

from atomic_datasets import datatypes
from atomic_datasets.datasets.platonic_solids import to_graph


class Tetris(datatypes.MolecularDataset):
    """Dataset of 3D Tetris shapes from from https://docs.e3nn.org/en/stable/examples/tetris_gate.html."""

    def __init__(
        self,
    ):
        super().__init__(atomic_numbers=[1])

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
        self.all_graphs = [to_graph(piece) for piece in tetris_pieces]

    def __len__(self) -> int:
        return len(self.all_graphs)

    def __getitem__(self, idx: int) -> datatypes.Graph:
        return self.all_graphs[idx]

