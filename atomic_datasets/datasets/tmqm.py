from typing import Iterable
import os

import tqdm
import numpy as np
import ase.io

from atomic_datasets import utils
from atomic_datasets import datatypes


TMQM_URL = r"https://github.com/bbskjelstad/tmqm.git"


class tmQMDataset(datatypes.MolecularDataset):
    """TMQM dataset."""

    def __init__(self, root_dir: str):
        super().__init__()
        
        if root_dir is None:
            raise ValueError("root_dir must be provided.")
            
        self.root_dir = root_dir
        self.preprocessed = False

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.arange(1, 81)

    def preprocess(self):
        preprocess(self.root_dir)
        self.preprocessed = True

    def __iter__(self) -> Iterable[datatypes.Graph]:
        if not self.preprocessed:
            self.preprocess()

        while True:
            yield from load_tmQM(self.root_dir)


def preprocess(root_dir: str):
    """Preprocess the files for the tmQM dataset."""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    xyzs_path = os.path.join(root_dir, "xyz")
    if os.path.exists(xyzs_path):
        print(f"Using downloaded data: {xyzs_path}")
    else:
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
    xyzs_path = os.path.join(root_dir, "xyz")
    for mol_file in tqdm.tqdm(sorted(os.listdir(xyzs_path))):
        mol_file = os.path.join(xyzs_path, mol_file)
        mol_as_ase = ase.io.read(mol_file, format="xyz")
        if mol_as_ase is None:
            continue
        yield utils.ase_atoms_to_graph(mol_as_ase)
