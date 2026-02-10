import numpy as np
from typing import Sequence, Iterable, Any

Graph = dict


class MolecularDataset:
    """A concrete dataset class for 3D molecular graphs."""

    def __init__(self, atomic_numbers: Sequence[int]):
        """
        Initialize the dataset with a specific atom mapping.
        
        Args:
            atomic_numbers: A sequence of atomic numbers for creating species. 
                            Example: [1, 6, 7] represents H, C, N atoms, and maps them to species indices 0, 1, 2 respectively.
        """
        self._species_to_atomic_map = np.asarray(atomic_numbers)
        max_z = np.max(self._species_to_atomic_map) + 1
        self._atomic_to_species_lookup = np.full(max_z, -1, dtype=np.int32)
        
        for species_id, z in enumerate(self._species_to_atomic_map):
            self._atomic_to_species_lookup[z] = species_id

    @property
    def num_species(self) -> int:
        """Total number of unique atom types in this dataset."""
        return len(self._species_to_atomic_map)

    def get_atomic_numbers(self) -> np.ndarray:
        """Returns the array of atomic numbers corresponding to species indices."""
        return self._species_to_atomic_map
    
    def atom_types(self) -> np.ndarray:
        """Returns the array of atomic symbols corresponding to species indices."""
        return np.array([utils.atomic_number_to_symbol(z) for z in self._species_to_atomic_map])
    
    def atomic_numbers_to_species(self, atomic_numbers: np.ndarray) -> np.ndarray:
        """
        Maps raw atomic numbers (e.g., 6, 8, 9) to species indices (e.g., 0, 1, 2).
        """
        species = self._atomic_to_species_lookup[atomic_numbers]
        if np.any(species == -1):
            invalid = np.unique(atomic_numbers[species == -1])
            raise ValueError(f"Unsupported atomic numbers found: {invalid}")
        return species

    def species_to_atomic_numbers(self, species: np.ndarray) -> np.ndarray:
        """Maps species indices back to raw atomic numbers."""
        return self._species_to_atomic_map[species]

    def __len__(self) -> int:
        raise NotImplementedError("This method should return the dataset size.")

    def __getitem__(self, idx: int) -> Graph:
        raise NotImplementedError("This method should return a single molecular graph.")
    
    def __iter__(self) -> Iterable[Graph]:
        """Iterate over all graphs in the dataset."""
        for i in range(len(self)):
            yield self[i]
