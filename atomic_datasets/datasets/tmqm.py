from typing import Iterable, Dict
import os

import tqdm
import numpy as np
import ase.io
from typing import Optional

from atomic_datasets import utils
from atomic_datasets import datatypes


TMQM_URL = r"https://github.com/bbskjelstad/tmqm.git"


class tmQM(datatypes.MolecularDataset):
    """TMQM dataset."""

    def __init__(
            self,
            root_dir: str,
            split: Optional[str] = None,
            start_index: Optional[int] = None,
            end_index: Optional[int] = None,
            rng_seed: int = 0,
        ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.preprocessed = False
        self.all_graphs = None
        self.split = split
        self.start_index = start_index
        self.end_index = end_index
        self.rng = np.random.default_rng(rng_seed)

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.arange(1, 81)
    
    @staticmethod
    def species_to_atomic_numbers() -> Dict[int, int]:
        return {i: i+1 for i in range(80)}

    def preprocess(self):
        self.preprocessed = True
        self.all_graphs = list(load_tmQM(self.root_dir))
        splits = self.split_indices()
        split = splits[self.split]
        if self.start_index is not None:
            split = split[self.start_index :]
        if self.end_index is not None:
            split = split[: self.end_index]
        self.all_graphs = [self.all_graphs[i] for i in split]
    
    def split_indices(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of indices for each split."""
        # if self.train_on_single_molecule:
        #     return {
        #         "train": [self.train_on_single_molecule_index],
        #         "val": [self.train_on_single_molecule_index],
        #         "test": [self.train_on_single_molecule_index],
        #     }

        num_train_molecules = 69000
        num_val_molecules = 9000
        num_test_molecules = 8665

        total_mols = num_train_molecules + num_val_molecules + num_test_molecules
        indices = np.arange(total_mols)
        self.rng.shuffle(indices)
        splits = {
            "train": np.arange(num_train_molecules),
            "val": np.arange(
                num_train_molecules,
                num_train_molecules + num_val_molecules,
            ),
            "test": np.arange(
                num_train_molecules + num_val_molecules,
                min(len(self.all_graphs), total_mols),
            ),
        }
        splits = {k: indices[v] for k, v in splits.items()}
        return splits

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


def get_raw_data(root_dir: str):
    """Preprocess the files for the tmQM dataset."""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    xyzs_path = os.path.join(root_dir, "xyz")
    if os.path.exists(xyzs_path) and len(os.listdir(xyzs_path)) > 0:
        print(f"Using downloaded data: {xyzs_path}")
        return

    print(f"Cloning tmQM repository to {root_dir}")
    _ = utils.clone_url(TMQM_URL, root_dir)
    os.makedirs(xyzs_path)

    for i in range(1, 3):
        gz_path = os.path.join(root_dir, "tmqm/tmQM", f"tmQM_X{i}.xyz.gz")

        mol_file = utils.extract_gz(gz_path)
        with open(mol_file, "r") as f:
            all_xyzs = f.read().split("\n\n")
            for xyz_n, xyz in enumerate(all_xyzs):
                if xyz == "":
                    continue
                xyz_lines = xyz.split("\n")
                assert len(xyz_lines) == int(xyz_lines[0]) + 2
                with open(os.path.join(xyzs_path, f"X{i}_{xyz_n}.xyz"), "w") as f:
                    f.write(xyz)


def load_tmQM(root_dir: str) -> Iterable[datatypes.Graph]:
    """Load the tmQM dataset."""
    get_raw_data(root_dir)
    xyzs_path = os.path.join(root_dir, "xyz")
    for mol_file in tqdm.tqdm(sorted(os.listdir(xyzs_path)), desc="Loading tmQM"):
        mol_file = os.path.join(xyzs_path, mol_file)
        mol_as_ase = ase.io.read(mol_file, format="xyz")
        if mol_as_ase is None:
            continue
        yield utils.ase_atoms_to_graph(mol_as_ase)
