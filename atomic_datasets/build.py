from typing import Dict
import numpy as np
import ml_collections

from atomic_datasets import dataset
from atomic_datasets.datasets import QM9Dataset, PlatonicSolidsDataset, GEOMDrugsDataset, TMQMDataset


def get_dataset(config: ml_collections.ConfigDict) -> dataset.InMemoryDataset:
    """Creates the dataset of structures, as specified in the config."""

    if config.dataset == "qm9":
        return QM9Dataset(
            root_dir=config.root_dir,
            check_molecule_sanity=config.get("check_molecule_sanity", False),
            use_edm_splits=config.use_edm_splits,
            num_train_molecules=config.num_train_molecules,
            num_val_molecules=config.num_val_molecules,
            num_test_molecules=config.num_test_molecules,
        )
    
    if config.dataset == "tmqm":
        return TMQMDataset(
            root_dir=config.root_dir,
            num_train_molecules=config.num_train_molecules,
            num_val_molecules=config.num_val_molecules,
            num_test_molecules=config.num_test_molecules,
        )

    if config.dataset == "platonic_solids":
        return PlatonicSolidsDataset(
            train_solids=config.train_solids,
            val_solids=config.val_solids,
            test_solids=config.test_solids,
        )

    if config.dataset == "geom_drugs":
        return GEOMDrugsDataset(
            root_dir=config.root_dir,
            use_gcdm_splits=config.use_gcdm_splits,
            num_train_molecules=config.num_train_molecules,
            num_val_molecules=config.num_val_molecules,
            num_test_molecules=config.num_test_molecules,
        )

    raise ValueError(
        f"Unknown dataset: {config.dataset}. Available datasets: qm9, platonic_solids, geom_drugs"
    )


def get_atomic_numbers(dataset: str) -> Dict[str, int]:
    """Returns a dictionary mapping atomic symbols to atomic numbers."""
    if dataset == "qm9":
        return QM9Dataset.get_atomic_numbers()
    elif dataset == "tmqm":
        return TMQMDataset.get_atomic_numbers()
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
    