from typing import Iterable, Dict, Optional, Tuple

import os
import logging

import numpy as np
import pandas as pd
import random
import rdkit.Chem as Chem
import tqdm

from atomic_datasets import utils
from atomic_datasets import datatypes

CROSSDOCKED_URL="https://ndownloader.figshare.com/articles/25878871/versions/3"
SPLIT_URL="https://drive.google.com/uc?export=download&id=1mycOKpphVBQjxEbpn1AwdpQs8tNVbxKY"

class CrossDocked(datatypes.MolecularDataset):
    """The CrossDocked dataset as used by https://proceedings.neurips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html."""

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        use_SBDD_splits: bool = False,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        train_on_single_molecule: Optional[bool] = False,
        train_on_single_molecule_index: Optional[int] = 0,
        use_cache: bool = True,  # New parameter to enable/disable caching
        cache_dir: Optional[str] = None,  # New parameter to specify cache directory
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.split = split
        self.use_SBDD_splits = use_SBDD_splits
        self.start_index = start_index
        self.end_index = end_index
        self.all_graphs = None
        self.preprocessed = False
        self.train_on_single_molecule = train_on_single_molecule
        self.train_on_single_molecule_index = train_on_single_molecule_index
        self.use_cache = use_cache
        self.cache_dir = cache_dir

    @classmethod
    def atom_types(cls) -> np.ndarray:
        return utils.atomic_numbers_to_symbols(cls.get_atomic_numbers())

    @classmethod
    def get_atomic_numbers(cls) -> np.ndarray:
        return np.asarray([1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 33, 34, 35, 37, 39, 42, 44, 45, 48, 50, 51, 53, 55, 56, 65, 70, 74, 78, 79, 80, 81, 82])

    def preprocess(self):
        self.preprocessed = True
        preprocess_directory(self.root_dir)

        if not self.use_SBDD_splits:
            if self.train_on_single_molecule:
                self.start_index = self.train_on_single_molecule_index
                self.end_index = self.train_on_single_molecule_index + 1

            # Use cached version if enabled
            if self.use_cache:
                load_crossdocked_fn = utils.cache_to_file("crossdocked", self.cache_dir)(load_crossdocked)
            else:
                load_crossdocked_fn = load_crossdocked

            all_graphs = list(
                load_crossdocked_fn(
                    self.root_dir,
                )
            )
            random.seed(0)
            random.shuffle(all_graphs)
            # if start_index/end_index are None, they default to the start/end of the list when used as indices
            self.all_graphs = all_graphs[self.start_index : self.end_index]
            return

        self.all_graphs = list(load_crossdocked(self.root_dir))
        self.all_graphs = np.array(self.all_graphs)
        train_test_splits = get_SBDD_splits(self.root_dir)
        split = []
        for i, graph in enumerate(self.all_graphs):
            if self.split == "val":
                if graph["properties"]["pocket_file"] in train_test_splits["train"]: continue
                if graph["properties"]["pocket_file"] in train_test_splits["test"]: continue
                split.append(i)
            if graph["properties"]["pocket_file"] in train_test_splits[self.split]:
                split.append(i)
            if len(split) == 100:  # TODO get a better way of making this split
                break
        # TODO further restrict the val set using clustering

        if self.start_index is not None:
            split = split[self.start_index :]
        if self.end_index is not None:
            split = split[: self.end_index]
        self.all_graphs = self.all_graphs[split]


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

    @classmethod
    def species_to_atomic_numbers(cls) -> Dict[int, int]:
        return {i: atomic_number for i, atomic_number in enumerate(CrossDocked.get_atomic_numbers())}

def preprocess_directory(root_dir: str) -> None:
    """Preprocess the files for the CrossDocked dataset."""
    raw_mols_path = os.path.join(root_dir, "crossdocked_pocket10_with_protein")
    if os.path.exists(raw_mols_path):
        print(f"Using downloaded data: {raw_mols_path}")
        return

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    print(f"Downloading CrossDocked dataset to {root_dir}")
    path = utils.download_url(CROSSDOCKED_URL, root_dir, "crossdocked.zip")
    utils.extract_zip(path, root_dir)
    utils.extract_gz(os.path.join(root_dir, "crossdocked_pocket10_with_protein.tar.gz"))
    utils.extract_tar(
        os.path.join(root_dir, "crossdocked_pocket10_with_protein.tar"),
        root_dir,
    )

    print(f"Downloading CrossDocked splits to {root_dir}")
    path = utils.download_url(SPLIT_URL, root_dir, "split_by_name.pt")

    print("Download complete.")

def load_crossdocked(
    root_dir: str,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> Iterable[datatypes.Graph]:
    """Load the CrossDocked2020 dataset."""

    data_dir = os.path.join(root_dir, "crossdocked_pocket10_with_protein")

    for pocket in tqdm.tqdm(os.listdir(data_dir), desc="Loading CrossDocked"):
        if not os.path.isdir(os.path.join(data_dir, pocket)):
            continue
        pocket_dir = os.path.join(data_dir, pocket)
        files = os.listdir(pocket_dir)
        pdb_ids = set(['_'.join(x.split('_')[:2]) for x in files])

        for pdb_id in pdb_ids:
            file_pairs = []
            for f in files:
                if not (len(pdb_id) < len(f) and f[:len(pdb_id)] == pdb_id):
                    continue
                if f[-4:] == ".sdf":
                    ligand_file = os.path.join(pocket, f)
                    pocket_file = ligand_file[:-4] + "_pocket10.pdb"
                    file_pairs.append(
                        (pocket_file, ligand_file)
                    )
                    continue

            for (pocket_file, ligand_file) in file_pairs:
                protein = Chem.MolFromPDBFile(
                    os.path.join(data_dir, pocket_file),
                    sanitize=False,
                    removeHs=False,
                )
                with Chem.SDMolSupplier(
                    os.path.join(data_dir, ligand_file),
                    sanitize=False,
                    removeHs=False,
                ) as suppl:
                    ligand = next(suppl)
                try:
                    target = Chem.CombineMols(protein, ligand)
                except Exception as e:
                    print("ligand file path:", os.path.join(data_dir, ligand_file))
                    print("ligand:", ligand)
                    print("files:", (pocket_file, ligand_file))
                    raise e

                atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in target.GetAtoms()])

                frag = datatypes.Graph(
                    nodes=dict(
                        positions=np.asarray(target.GetConformer().GetPositions()),
                        species=CrossDocked.atomic_numbers_to_species(atomic_numbers),
                        atom_types=utils.atomic_numbers_to_symbols(atomic_numbers),
                    ),
                    edges=None,
                    receivers=None,
                    senders=None,
                    globals=None,
                    n_node=np.asarray([target.GetNumAtoms()]),
                    n_edge=None,
                    properties=dict(
                        starting_fragment_mask=np.array(
                            [1] * protein.GetNumAtoms() + [0] * ligand.GetNumAtoms(), dtype=bool
                        ),
                        pocket_file=pocket_file,
                        mol=target,
                    ),
                )
                yield frag

def get_SBDD_splits(
    root_dir: str,
) -> Dict[str, np.ndarray]:
    """Use splits from Luo et al. (https://proceedings.neurips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html)."""
    import torch

    train_test_splits = torch.load(os.path.join(root_dir, "split_by_name.pt"))
    splits = {split: [x[0] for x in train_test_splits[split]] for split in train_test_splits}

    return splits
