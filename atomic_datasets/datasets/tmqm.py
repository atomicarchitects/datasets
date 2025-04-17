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
    """The tmQM dataset from https://pubs.acs.org/doi/10.1021/acs.jcim.0c01041."""

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        use_default_splits: bool = True,
        splits: Optional[Dict[str, np.ndarray]] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        rng_seed: int = 0,
        train_on_single_molecule: Optional[bool] = False,
        train_on_single_molecule_index: Optional[int] = 0,
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

        if use_default_splits:
            if splits is not None:
                raise ValueError("If use_default_splits is True, splits must be None.")
            self.splits = {
                "train": np.arange(69000),
                "val": np.arange(69000, 78000),
                "test": np.arange(78000, 86665),
            }
        else:
            if splits is None:
                raise ValueError(
                    "If use_default_splits is False, splits must be provided."
                )
            self.splits = splits

        self.rng = np.random.default_rng(rng_seed)
        self.train_on_single_molecule = train_on_single_molecule
        self.train_on_single_molecule_index = train_on_single_molecule_index

    @classmethod
    def atom_types(cls) -> np.ndarray:
        return utils.atomic_numbers_to_symbols(cls.get_atomic_numbers())

    @classmethod
    def get_atomic_numbers(cls) -> np.ndarray:
        return np.arange(1, 81)

    @classmethod
    def species_to_atomic_numbers(cls) -> Dict[int, int]:
        return {i: i + 1 for i in range(80)}

    def preprocess(self):
        self.preprocessed = True
        self.all_graphs = list(
            load_tmQM(
                self.root_dir,
            )
        )

        if self.split is None:
            return

        splits = self.split_indices()
        split = splits[self.split]
        if self.start_index is not None:
            split = split[self.start_index :]
        if self.end_index is not None:
            split = split[: self.end_index]
        self.all_graphs = [self.all_graphs[i] for i in split]

    def split_indices(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of indices for each split."""
        if self.train_on_single_molecule:
            return {
                "train": [self.train_on_single_molecule_index],
                "val": [self.train_on_single_molecule_index],
                "test": [self.train_on_single_molecule_index],
            }

        indices = np.arange(sum(len(v) for v in self.splits.values()))
        self.rng.shuffle(indices)
        splits = {k: indices[v] for k, v in self.splits.items()}
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


def download_data(root_dir: str) -> str:
    """Preprocess the files for the tmQM dataset."""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    xyzs_path = os.path.join(root_dir, "xyz")
    if os.path.exists(xyzs_path) and len(os.listdir(xyzs_path)) > 0:
        print(f"Using downloaded data: {xyzs_path}")
        return xyzs_path

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

    return xyzs_path


def load_tmQM(
    root_dir: str,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> Iterable[datatypes.Graph]:
    """Load the tmQM dataset."""
    xyzs_path = download_data(root_dir)

    for index, mol_file in enumerate(
        tqdm.tqdm(sorted(os.listdir(xyzs_path)), desc="Loading tmQM")
    ):
        if start_index is not None and index < start_index:
            continue

        if end_index is not None and index >= end_index:
            break

        mol_file = os.path.join(xyzs_path, mol_file)
        mol_as_ase = ase.io.read(mol_file, format="xyz")
        if mol_as_ase is None:
            continue

        yield datatypes.Graph(
            nodes=dict(
                positions=np.asarray(mol_as_ase.positions),
                species=mol_as_ase.numbers - 1,
                atom_types=utils.atomic_numbers_to_symbols(mol_as_ase.numbers),
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=None,
            n_node=np.asarray([len(mol_as_ase.numbers)]),
            n_edge=None,
        )
