from typing import Iterable, Dict, Sequence
import abc

import jraph


class InMemoryDataset(abc.ABC):
    """Abstract base class for in-memory datasets."""

    def num_species(self) -> int:
        """Return the number of atom types."""
        return len(self.get_atomic_numbers())

    @abc.abstractmethod
    def get_atomic_numbers(self) -> Sequence[int]:
        """Returns a sorted list of the atomic numbers observed in the dataset."""

    @abc.abstractmethod
    def structures(self) -> Iterable[jraph.GraphsTuple]:
        """Return a list of all completed structures."""

    @abc.abstractmethod
    def split_indices(self) -> Dict[str, Sequence[int]]:
        """Return a dictionary of split indices."""
        pass
