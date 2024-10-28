from typing import List, Iterable, Dict, Set
import logging
import os

import tqdm
import sh
import numpy as np
import ase
import jraph


TMQM_URL = "https://github.com/bbskjelstad/tmqm.git"


def _molecule_to_structure(molecule: ase.Atoms) -> jraph.GraphsTuple:
    """Converts a molecule to a jraph.GraphsTuple object."""
    return jraph.GraphsTuple(
        nodes=dict(
            positions=np.asarray(molecule.positions),
            species=molecule.numbers - 1
        ),
        edges=None,
        receivers=None,
        senders=None,
        globals=None,
        n_node=np.asarray([len(molecule.numbers)]),
        n_edge=None,
    )


class TMQMDataset(InMemoryDataset):
    """TMQM dataset."""

    def __init__(self, root_dir: str, num_train_molecules: int, 
                 num_val_molecules: int, num_test_molecules: int):
        super().__init__()
        
        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        if num_train_molecules is None or num_val_molecules is None or num_test_molecules is None:
            raise ValueError("num_train_molecules, num_val_molecules, and num_test_molecules must be provided.")
            
        self.root_dir = root_dir
        self.num_train_molecules = num_train_molecules
        self.num_val_molecules = num_val_molecules
        self.num_test_molecules = num_test_molecules

        self.all_structures = None

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.arange(1, 81)

    def structures(self) -> Iterable[jraph.GraphsTuple]:
        if self.all_structures == None:
            self.all_structures = load_tmqm(self.root_dir)
            logging.info("Loaded TMQM dataset.")
        return self.all_structures 

    @staticmethod
    def species_to_atom_types() -> Dict[int, str]:
        ptable = PeriodicTable()
        return {
            i: ptable.get_symbol(i) for i in range(80)
        }

    def split_indices(self) -> Dict[str, Set[int]]:
        # Create a random permutation of the indices.
        np.random.seed(0)
        indices = np.random.permutation(86665)
        permuted_indices = {
            "train": indices[:self.num_train_molecules],
            "val": indices[self.num_train_molecules:self.num_train_molecules + self.num_val_molecules],
            "test": indices[self.num_train_molecules + self.num_val_molecules:self.num_train_molecules + self.num_val_molecules + self.num_test_molecules],
        }
        return permuted_indices


def load_tmqm(root_dir: str) -> List[ase.Atoms]:
    """Load the TMQM dataset."""
    mols = []
    data_path = root_dir
    xyzs_path = os.path.join(data_path, "xyz")
    if os.path.exists(xyzs_path):
        logging.info(f"Using downloaded data: {xyzs_path}")
    else:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        logging.info(f"Cloning TMQM repository to {root_dir}...")
        _ = utils.clone_url(TMQM_URL, root_dir)
        if not os.path.exists(xyzs_path):
            os.makedirs(xyzs_path)

        for i in range(1, 3):
            gz_path = os.path.join(data_path, "tmqm/data", f"tmQM_X{i}.xyz.gz")
            logging.info(f"Unzipping {gz_path}...")
            sh.gunzip(gz_path)

            mol_file = os.path.join(data_path, "tmqm/data", f"tmQM_X{i}.xyz")
            with open(mol_file, "r") as f:
                all_xyzs = f.read().split("\n\n")
                for xyz_n, xyz in enumerate(all_xyzs):
                    if xyz == "":
                        continue
                    xyz_lines = xyz.split("\n")
                    assert len(xyz_lines) == int(xyz_lines[0]) + 2
                    with open(os.path.join(xyzs_path, f"X{i}_{xyz_n}.xyz"), "w") as f:
                        f.write(xyz)

    for mol_file in tqdm.tqdm(os.listdir(xyzs_path)):
        mol_as_ase = ase.io.read(os.path.join(xyzs_path, mol_file), format="xyz")
        if mol_as_ase is None:
            continue
        mols.append(_molecule_to_structure(mol_as_ase))

    logging.info(f"Loaded {len(mols)} molecules.")
    return mols
