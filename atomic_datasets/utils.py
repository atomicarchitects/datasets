from typing import Dict
import numpy as np

from atomic_datasets.datasets.qm9 import QM9Dataset
from atomic_datasets.datasets.geom_drugs import GEOMDrugsDataset
from atomic_datasets.datasets.platonic_solids import PlatonicSolidsDataset
from atomic_datasets.datasets.tmqm import tmQMDataset


def get_atomic_numbers(dataset: str) -> Dict[str, int]:
    """Returns a dictionary mapping atomic symbols to atomic numbers."""
    if dataset == "qm9":
        return QM9Dataset.get_atomic_numbers()
    elif dataset == "tmqm":
        return tmQMDataset.get_atomic_numbers()
    elif dataset == "platonic_solids":
        return PlatonicSolidsDataset.get_atomic_numbers()
    elif dataset == "geom_drugs":
        return GEOMDrugsDataset.get_atomic_numbers()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def species_to_atomic_numbers(
    species: np.ndarray, dataset: str,
) -> np.ndarray:
    """Returns the atomic numbers for the species."""
    atomic_numbers = get_atomic_numbers(dataset)
    return np.asarray(atomic_numbers)[species]
    