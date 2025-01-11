from typing import Dict, Iterable
import os

import numpy as np
import tqdm

from atomic_datasets import datatypes
from atomic_datasets import utils


GEOM_DRUGS_URL = r"https://zenodo.org/records/7881981/files/EDM.tar.gz"


class GEOMDrugsDataset(datatypes.MolecularDataset):
    """GEOM (Drugs) dataset."""

    def __init__(
        self,
        root_dir: str,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.preprocessed = False
        self.all_graphs = None

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83])

    def preprocess(self):
        self.preprocessed = True

        preprocess(self.root_dir)
        self.all_graphs = list(load_GEOM_drugs(self.root_dir))

    @utils.after_preprocess
    def __iter__(self) -> Iterable[datatypes.Graph]:
        for graph in self.all_graphs:
            yield graph
    
    @utils.after_preprocess
    def __len__(self) -> int:
        return len(self.all_graphs)

    @utils.after_preprocess
    def __getitem__(self, idx: int) -> datatypes.Graph:
        return self.all_graphs[idx]


def preprocess(root_dir: str):
    """Preprocess the files for the GEOM (Drugs) dataset."""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    print(f"Downloading GEOM (Drugs) dataset to {root_dir}")
    path = utils.download_url(GEOM_DRUGS_URL, root_dir)
    path = utils.extract_gz(path)
    path = utils.extract_tar(path, root_dir)
    print("Download complete.")


def load_GEOM_drugs(root_dir: str) -> Iterable[datatypes.Graph]:
    """Adapted from https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/src/datamodules/components/edm/build_geom_dataset.py."""

    conformation_file = os.path.join(root_dir, "GEOM_drugs_30.npy")
    all_data = np.load(conformation_file)  # 2D array: num_atoms x 5

    mol_id = all_data[:, 0].astype(int)
    conformers = all_data[:, 1:]
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)

    for datum in tqdm.tqdm(data_list, desc="Loading GEOM (Drugs)"):
        atom_types = datum[:, 0].astype(int)
        atom_positions = datum[:, 1:].astype(float)
        species = GEOMDrugsDataset.atomic_numbers_to_species(atom_types)

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
    """Splits for GEOM (Drugs). Adapted from https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/src/datamodules/components/edm/build_geom_dataset.py."""

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
