from typing import NamedTuple, Dict, Optional, Iterable, Sequence
import abc

import jraph
import jax.numpy as jnp


MolecularGraph = jraph.GraphsTuple


class InMemoryMolecularDataset(abc.ABC):
    """Abstract base class for in-memory molecular datasets."""

    def num_species(self) -> int:
        """Return the number of atom types."""
        return len(self.get_atomic_numbers())

    @abc.abstractmethod
    def get_atomic_numbers(self) -> Sequence[int]:
        """Returns a sorted list of the atomic numbers observed in the dataset."""

    @abc.abstractmethod
    def __iter__(self) -> Iterable[MolecularGraph]:
        """Return an iterator over the dataset."""
        pass

    @abc.abstractmethod
    def split_indices(self) -> Dict[str, Sequence[int]]:
        """Return a dictionary of split indices."""
        pass
