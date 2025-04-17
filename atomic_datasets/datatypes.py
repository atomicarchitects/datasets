from typing import Dict, Iterable, Sequence, Any
import abc

import numpy as np

Graph = dict


class MolecularDataset(abc.ABC):
    """Abstract base class for in-memory molecular datasets."""

    def num_species(self) -> int:
        """Return the number of atom types."""
        return len(self.get_atomic_numbers())

    @abc.abstractmethod
    def atom_types(self) -> Sequence[str]:
        """Return all possible atom types."""

    @abc.abstractmethod
    def get_atomic_numbers(self) -> Sequence[int]:
        """Returns a sorted list of the atomic numbers observed in the dataset."""

    @classmethod
    def species_to_atomic_numbers(cls) -> np.ndarray:
        """Returns the atomic numbers for the species."""

    @classmethod
    def atomic_numbers_to_species(cls, atomic_numbers: np.ndarray) -> np.ndarray:
        """Returns the species for the atomic numbers."""
        all_atomic_numbers = cls.get_atomic_numbers()
        return np.searchsorted(all_atomic_numbers, atomic_numbers)

    @abc.abstractmethod
    def __iter__(self) -> Iterable[Graph]:
        """Return an iterator over the dataset."""
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Graph:
        """Return the graph at the specified index."""
        pass
