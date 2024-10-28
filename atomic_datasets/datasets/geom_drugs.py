from typing import Dict, Iterable, List
import os

import logging
import numpy as np
import jraph

from atomic_datasets import InMemoryDataset


class GEOMDrugsDataset(InMemoryDataset):
    def __init__(
        self,
        root_dir: str,
        use_gcdm_splits: bool,
        num_train_molecules: int,
        num_val_molecules: int,
        num_test_molecules: int,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.use_gcdm_splits = use_gcdm_splits
        self.num_train_molecules = num_train_molecules
        self.num_val_molecules = num_val_molecules
        self.num_test_molecules = num_test_molecules
        self.all_structures = None

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83])

    def structures(self) -> Iterable[jraph.GraphsTuple]:
        if self.all_structures is None:
            self.all_structures = load_geom_drugs(self.root_dir)
        return self.all_structures

    def split_indices(self) -> Dict[str, np.ndarray[int]]:
        """Returns the indices for the train, val, and test splits."""
        if not self.use_gcdm_splits:
            raise NotImplementedError("Only GCDM splits are supported for QM9.")

        splits = get_gcdm_splits(self.root_dir)
        requested_splits = {
            "train": self.num_train_molecules,
            "val": self.num_val_molecules,
            "test": self.num_test_molecules,
        }
        for split_name, num_molecules in requested_splits.items():
            original_split_size = len(splits[split_name])
            if num_molecules > original_split_size:
                raise ValueError(
                    f"Requested {num_molecules} molecules for split {split_name}, but only {original_split_size} are available."
                )
            logging.info(
                f"Using {num_molecules} molecules out of {original_split_size} in split {split_name}."
            )
            splits[split_name] = splits[split_name][:num_molecules]
        return splits


def load_geom_drugs(root_dir: str) -> List[jraph.GraphsTuple]:
    """Adapted from https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/atomic_datasets/datamodules/components/edm/build_geom_dataset.py."""

    conformation_file = os.path.join(root_dir, "GEOM_drugs_30.npy")
    all_data = np.load(conformation_file)  # 2D array: num_atoms x 5

    mol_id = all_data[:, 0].astype(int)
    conformers = all_data[:, 1:]
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)
    atomic_numbers = np.asarray(GEOMDrugsDataset.get_atomic_numbers())

    all_structures = []
    for datum in data_list:
        atom_types = datum[:, 0].astype(int)
        atom_positions = datum[:, 1:].astype(float)
        species = atomic_numbers.searchsorted(atom_types)

        structure = jraph.GraphsTuple(
            nodes=dict(positions=atom_positions, species=species),
            edges=None,
            senders=None,
            receivers=None,
            n_edge=None,
            n_node=np.array([len(atom_types)]),
            globals=None,
        )
        all_structures.append(structure)

    return all_structures


def get_gcdm_splits(root_dir: str) -> Dict[str, np.ndarray]:
    """Adapted from https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/atomic_datasets/datamodules/components/edm/build_geom_dataset.py."""

    permutation_file = os.path.join(root_dir, "GEOM_permutation.npy")
    permutation = np.load(permutation_file)

    num_mol = len(permutation)
    val_proportion = 0.1
    val_split = int(num_mol * val_proportion)
    test_proportion = 0.1
    test_split = val_split + int(num_mol * test_proportion)
    val_indices, test_indices, train_indices = np.split(
        permutation, [val_split, test_split]
    )

    return {"train": train_indices, "val": val_indices, "test": test_indices}
