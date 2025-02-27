"""Module to store information about the periodic table (groups, rows, blocks)."""

import numpy as np


class PeriodicTable:
    """Class to store information about the periodic table (groups, rows, blocks)."""

    def __init__(self):
        self.groups = np.array(
            [
                0,
                17,
            ]
            + [0, 1, 12, 13, 14, 15, 16, 17] * 2
            + list(range(0, 18)) * 2
            + [0, 1]
            + [2] * 15
            + list(range(3, 18))
            + [0, 1]
            + [2] * 15
            + list(range(3, 18))
        )
        self.rows = np.array(
            [0] * 2 + [1] * 8 + [2] * 8 + [3] * 18 + [4] * 18 + [5] * 32 + [6] * 32
        )
        # s = 0, p = 1,
        self.blocks = np.array(
            [0] * 2
            + [0] * 2
            + [1] * 6
            + [0] * 2
            + [1] * 6
            + [0] * 2
            + [2] * 10
            + [1] * 6
            + [0] * 2
            + [2] * 10
            + [1] * 6
            + [0] * 2
            + [3] * 14
            + [2] * 10
            + [1] * 6
            + [0] * 2
            + [3] * 14
            + [2] * 10
            + [1] * 6
        )

        self.symbols = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Nh",
            "Fl",
            "Mc",
            "Lv",
            "Ts",
            "Og",
        ]

        self.radii = [
            0.79,
            0.49,
            2.1,
            1.4,
            1.2,
            0.91,
            0.75,
            0.65,
            0.57,
            0.51,
            2.2,
            1.7,
            1.8,
            1.5,
            1.2,
            1.1,
            0.97,
            0.88,
            2.8,
            2.2,
            2.1,
            2.0,
            1.9,
            1.9,
            1.8,
            1.7,
            1.7,
            1.6,
            1.6,
            1.5,
            1.8,
            1.5,
            1.3,
            1.2,
            1.1,
            1.0,
            3.0,
            2.5,
            2.3,
            2.2,
            2.1,
            2.0,
            2.0,
            1.9,
            1.8,
            1.8,
            1.8,
            1.7,
            2.0,
            1.7,
            1.5,
            1.4,
            1.3,
            1.2,
            3.3,
            2.8,
            2.7,
            2.7,
            2.7,
            2.6,
            2.6,
            2.6,
            2.6,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.4,
            2.4,
            2.3,
            2.2,
            2.1,
            2.0,
            2.0,
            1.9,
            1.9,
            1.8,
            1.8,
            1.8,
            2.1,
            1.8,
            1.6,
            1.5,
            1.4,
            1.3,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
        ]

    def get_group(self, atomic_number: int | np.ndarray) -> np.ndarray:
        return self.groups[atomic_number]

    def get_row(self, atomic_number: int | np.ndarray) -> np.ndarray:
        return self.rows[atomic_number]

    def get_block(self, atomic_number: int | np.ndarray) -> np.ndarray:
        return self.blocks[atomic_number]

    def get_symbol(self, atomic_number: int | np.ndarray) -> np.ndarray:
        return self.symbols[atomic_number]

    def get_radius(self, atomic_number: int | np.ndarray) -> np.ndarray:
        return self.radii[atomic_number]
