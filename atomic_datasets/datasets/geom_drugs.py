from typing import Dict, Iterable, List
import os

import logging
import numpy as np

from atomic_datasets import datatypes


class GEOMDrugsDataset(datatypes.InMemoryMolecularDataset):
    """GEOM (Drugs) dataset."""

    def __init__(
        self,
        root_dir: str,
    ):
        super().__init__()
        self.root_dir = root_dir

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83])

    def __iter__(self) -> Iterable[datatypes.Graph]:
        while True:
            yield from load_GEOM_DRUGS(self.root_dir)


def load_GEOM_DRUGS(root_dir: str) -> Iterable[datatypes.Graph]:
    """Adapted from https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/atomic_datasets/datamodules/components/edm/build_geom_dataset.py."""

    conformation_file = os.path.join(root_dir, "GEOM_drugs_30.npy")
    all_data = np.load(conformation_file)  # 2D array: num_atoms x 5

    mol_id = all_data[:, 0].astype(int)
    conformers = all_data[:, 1:]
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)
    atomic_numbers = np.asarray(GEOMDrugsDataset.get_atomic_numbers())

    for datum in data_list:
        atom_types = datum[:, 0].astype(int)
        atom_positions = datum[:, 1:].astype(float)
        species = atomic_numbers.searchsorted(atom_types)

        yield datatypes.Graph(
            nodes=dict(positions=atom_positions, species=species),
            edges=None,
            senders=None,
            receivers=None,
            n_edge=None,
            n_node=np.array([len(atom_types)]),
            globals=None,
        )


def get_GCDM_splits(root_dir: str) -> Dict[str, np.ndarray]:
    """Splits for GEOM (Drugs). Adapted from https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/atomic_datasets/datamodules/components/edm/build_geom_dataset.py."""

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
