from typing import Dict, Iterable, Optional
import os
import logging

import numpy as np
import tqdm

from atomic_datasets import datatypes
from atomic_datasets import utils


GEOM_DRUGS_URL = r"https://zenodo.org/records/7881981/files/EDM.tar.gz"


class GEOMDrugs(datatypes.MolecularDataset):
    """GEOM (Drugs) dataset."""

    def __init__(
        self,
        root_dir: str,
        use_GCDM_splits: bool = False,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.use_GCDM_splits = use_GCDM_splits
        self.start_index = start_index
        self.end_index = end_index

        self.preprocessed = False
        self.all_graphs = None


        if self.use_GCDM_splits:
            if self.split is None:
                raise ValueError("When use_GCDM_splits is True, split must be provided.")

            if self.start_index is not None or self.end_index is not None:
                logging.warning("When use_GCDM_splits is True, start_index and end_index refer to the indices of the GCDM splits.")

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83])

    def preprocess(self):
        self.preprocessed = True

        preprocess(self.root_dir)
        if not self.use_GCDM_splits:
            self.all_graphs = list(load_GEOM_drugs(self.root_dir, self.start_index, self.end_index))
            return

        self.all_graphs = list(load_GEOM_drugs(self.root_dir))
        splits = get_GCDM_splits(self.root_dir)
        split = splits[self.split]
        if self.start_index is not None:
            split = split[self.start_index:]
        if self.end_index is not None:
            split = split[:self.end_index]
        self.all_graphs = [self.all_graphs[i] for i in split]

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
    conformation_file = os.path.join(root_dir, "GEOM_drugs_30.npy")
    if os.path.exists(conformation_file):
        print(f"Using downloaded data: {conformation_file}")
        return

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    print(f"Downloading GEOM (Drugs) dataset to {root_dir}")
    path = utils.download_url(GEOM_DRUGS_URL, root_dir)
    path = utils.extract_gz(path)
    path = utils.extract_tar(path, root_dir)
    print("Download complete.")


def load_GEOM_drugs(root_dir: str, start_index: Optional[int], end_index: Optional[int]) -> Iterable[datatypes.Graph]:
    """Adapted from https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/src/datamodules/components/edm/build_geom_dataset.py."""

    conformation_file = os.path.join(root_dir, "GEOM_drugs_30.npy")
    all_data = np.load(conformation_file)  # 2D array: num_atoms x 5

    mol_id = all_data[:, 0].astype(int)
    conformers = all_data[:, 1:]
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)

    for index, datum in enumerate(tqdm.tqdm(data_list, desc="Loading GEOM (Drugs)")):
        if start_index is not None and index < start_index:
            continue

        if end_index is not None and index >= end_index:
            break

        atom_types = datum[:, 0].astype(int)
        atom_positions = datum[:, 1:].astype(float)
        species = GEOMDrugs.atomic_numbers_to_species(atom_types)

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
