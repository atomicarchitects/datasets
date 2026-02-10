"""Module to store information about the periodic table (groups, rows, blocks)."""

import numpy as np
from rdkit import Chem


class PeriodicTable:
    """Class to store information about the periodic table (groups, rows, blocks)."""

    def __init__(self):
        self.pt = Chem.GetPeriodicTable()
        
        # For blocks, we'll define them by atomic number ranges
        # s-block: groups 1-2, p-block: groups 13-18, d-block: groups 3-12, f-block: lanthanides/actinides
        self.blocks = np.array(
            [0] * 2  # H, He (s-block, though He is sometimes classified differently)
            + [0] * 2 + [1] * 6  # Period 2: Li-Be (s), B-Ne (p)
            + [0] * 2 + [1] * 6  # Period 3: Na-Mg (s), Al-Ar (p)
            + [0] * 2 + [2] * 10 + [1] * 6  # Period 4: K-Ca (s), Sc-Zn (d), Ga-Kr (p)
            + [0] * 2 + [2] * 10 + [1] * 6  # Period 5: Rb-Sr (s), Y-Cd (d), In-Xe (p)
            + [0] * 2 + [3] * 14 + [2] * 10 + [1] * 6  # Period 6: Cs-Ba (s), La-Yb (f), Hf-Hg (d), Tl-Rn (p)
            + [0] * 2 + [3] * 14 + [2] * 10 + [1] * 6  # Period 7: Fr-Ra (s), Ac-No (f), Rf-Cn (d), Nh-Og (p)
        )
        
        # Groups for elements (0-17 indexing)
        # Note: Lanthanides and actinides are typically assigned to group 3
        self.groups = np.array(
            [0, 17]  # H, He
            + [0, 1, 12, 13, 14, 15, 16, 17] * 2  # Periods 2-3
            + list(range(0, 18)) * 2  # Periods 4-5
            + [0, 1] + [2] * 15 + list(range(3, 18))  # Period 6 (La-Lu in group 2 placeholder)
            + [0, 1] + [2] * 15 + list(range(3, 18))  # Period 7 (Ac-Lr in group 2 placeholder)
        )

    def get_group(self, atomic_number: int | np.ndarray) -> int | np.ndarray:
        """Get the group number (0-17) for given atomic number(s)."""
        return self.groups[atomic_number]

    def get_row(self, atomic_number: int | np.ndarray) -> int | np.ndarray:
        """Get the period/row (0-indexed) for given atomic number(s)."""
        if isinstance(atomic_number, np.ndarray):
            return np.array([self.pt.GetRow(int(z)) for z in atomic_number])
        return self.pt.GetRow(atomic_number)

    def get_block(self, atomic_number: int | np.ndarray) -> int | np.ndarray:
        """Get the block (0=s, 1=p, 2=d, 3=f) for given atomic number(s)."""
        return self.blocks[atomic_number]

    def get_symbol(self, atomic_number: int | np.ndarray) -> str | list:
        """Get the element symbol for given atomic number(s)."""
        if isinstance(atomic_number, np.ndarray):
            return [self.pt.GetElementSymbol(int(z)) for z in atomic_number]
        return self.pt.GetElementSymbol(atomic_number)

    def get_radius(self, atomic_number: int | np.ndarray) -> float | np.ndarray:
        """Get the van der Waals radius for given atomic number(s)."""
        if isinstance(atomic_number, np.ndarray):
            return np.array([self.pt.GetRvdw(int(z)) for z in atomic_number])
        return self.pt.GetRvdw(atomic_number)