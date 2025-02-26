from typing import Iterable, Dict, Optional

import os
import logging

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import tqdm

from atomic_datasets import utils
from atomic_datasets import datatypes

QM9_URL = r"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"


class QM9(datatypes.MolecularDataset):
    """QM9 dataset."""

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        use_Anderson_splits: bool = False,
        check_with_rdkit: bool = False,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.split = split
        self.check_with_rdkit = check_with_rdkit
        self.use_Anderson_splits = use_Anderson_splits
        self.start_index = start_index
        self.end_index = end_index
        self.all_graphs = None
        self.preprocessed = False

        if self.use_Anderson_splits:
            if self.split is None:
                raise ValueError("When use_Anderson_splits is True, split must be provided.")

            if self.check_with_rdkit:
                raise ValueError("Splits determined by Anderson are not compatible with checking with RDKit.")

            if self.start_index is not None or self.end_index is not None:
                logging.warning("When use_Anderson_splits is True, start_index and end_index refer to the indices of the Anderson splits.")

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1, 6, 7, 8, 9])

    def preprocess(self):
        self.preprocessed = True

        preprocess_directory(self.root_dir)

        README = os.path.join(self.root_dir, "QM9_README")
        with open(README) as f:
            print("Dataset description:", f.read())

        if not self.use_Anderson_splits:
            self.all_graphs = list(load_qm9(self.root_dir, self.check_with_rdkit, self.start_index, self.end_index))
            return

        self.all_graphs = list(load_qm9(self.root_dir, self.check_with_rdkit))
        splits = get_Anderson_splits(self.root_dir)
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


def preprocess_directory(root_dir: str) -> None:
    """Preprocess the files for the QM9 dataset."""
    raw_mols_path = os.path.join(root_dir, "gdb9.sdf")
    if os.path.exists(raw_mols_path):
        print(f"Using downloaded data: {raw_mols_path}")
        return

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    print(f"Downloading QM9 dataset to {root_dir}")
    path = utils.download_url(QM9_URL, root_dir)
    utils.extract_zip(path, root_dir)
    print("Download complete.")


def load_qm9(
    root_dir: str,
    check_with_rdkit: bool = True,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> Iterable[datatypes.Graph]:
    """Load the QM9 dataset."""

    raw_mols_path = os.path.join(root_dir, "gdb9.sdf")
    supplier = Chem.SDMolSupplier(raw_mols_path, removeHs=False, sanitize=False)

    properties_csv_path = os.path.join(root_dir, "gdb9.sdf.csv")
    properties = pd.read_csv(properties_csv_path)
    properties.set_index("mol_id", inplace=True)

    for index, mol in enumerate(tqdm.tqdm(supplier, desc="Loading QM9")):
        if start_index is not None and index < start_index:
            continue

        if end_index is not None and index >= end_index:
            break

        if mol is None:
            raise ValueError("Failed to load molecule.")

        # Check that the molecule passes some basic checks from Posebusters.
        if check_with_rdkit and not utils.is_molecule_sane(mol):
            continue

        mol_id = mol.GetProp("_Name")
        mol_properties = properties.loc[mol_id].to_dict()
        mol_properties["mol_id"] = mol_id

        atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])

        yield datatypes.Graph(
            nodes=dict(
                positions=np.asarray(mol.GetConformer().GetPositions()),
                species=QM9.atomic_numbers_to_species(atomic_numbers)
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=None,
            n_node=np.asarray([mol.GetNumAtoms()]),
            n_edge=None,
            properties=mol_properties,
        )


def remove_uncharacterized_molecules(
    root_dir: str,
):
    """Remove molecules from the QM9 dataset that are uncharacterized."""
    def is_int(string: str) -> bool:
        try:
            int(string)
            return True
        except:
            return False

    print("Dropping uncharacterized molecules.")
    gdb9_url_excluded = "https://springernature.figshare.com/ndownloader/files/3195404"
    gdb9_txt_excluded = utils.download_url(gdb9_url_excluded, root_dir)

    # First, get list of excluded indices.
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0] for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert (
        len(excluded_idxs) == 3054
    ), f"There should be exactly 3054 excluded molecule. Found {len(excluded_idxs)}"

    # Cleanup file.
    try:
        os.remove(gdb9_txt_excluded)
    except OSError:
        pass

    # Now, create a list of included indices.
    Ngdb9 = 133885
    included_idxs = np.array(sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))
    return included_idxs, excluded_idxs


def get_Anderson_splits(
    root_dir: str,
) -> Dict[str, np.ndarray]:
    """Use splits from Anderson, et al. (https://arxiv.org/abs/1906.04015).

    Adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9/data/prepare/qm9.py.
    """
    included_idxs, excluded_idxs = remove_uncharacterized_molecules(root_dir)

    # Now, generate random permutations to assign molecules to training/valation/test sets.
    Nmols = len(included_idxs)
    Ntrain = 100000
    Ntest = int(0.1 * Nmols)
    Nval = Nmols - (Ntrain + Ntest)

    # Generate random permutation.
    np.random.seed(0)
    data_permutation = np.random.permutation(Nmols)

    train, val, test, extra = np.split(
        data_permutation, [Ntrain, Ntrain + Nval, Ntrain + Nval + Ntest]
    )

    assert len(extra) == 0, f"Split was inexact {len(train)} {len(val)} {len(test)} with {len(extra)} extra."

    train = included_idxs[train]
    val = included_idxs[val]
    test = included_idxs[test]

    splits = {"train": train, "val": val, "test": test}
    return splits
