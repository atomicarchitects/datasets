from typing import Iterable, Dict, Optional

import os

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import tqdm

from atomic_datasets import utils
from atomic_datasets import datatypes

QM9_URL = r"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"


class QM9Dataset(datatypes.MolecularDataset):
    """QM9 dataset."""

    def __init__(
        self,
        root_dir: str,
        check_with_rdkit: bool = False,
        remove_uncharacterized_molecules: bool = True,
        max_num_molecules: int = None,
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.check_with_rdkit = check_with_rdkit
        self.remove_uncharacterized_molecules = remove_uncharacterized_molecules
        self.max_num_molecules = max_num_molecules
        self.all_graphs = None
        self.preprocessed = False

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1, 6, 7, 8, 9])

    def preprocess(self):
        self.preprocessed = True

        preprocess(self.root_dir)
        self.all_graphs = list(load_qm9(self.root_dir, self.check_with_rdkit, self.max_num_molecules))

        if self.remove_uncharacterized_molecules:
            included_idxs, _ = remove_uncharacterized_molecules(self.root_dir)
            self.all_graphs = [self.all_graphs[i] for i in included_idxs if i < len(self.all_graphs)]

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

    readme = os.path.join(root_dir, "QM9_README")
    with open(readme) as f:
        print("Dataset description:", f.read())


def load_qm9(
    root_dir: str,
    check_with_rdkit: bool = True,
    max_num_molecules: Optional[int] = None,
) -> Iterable[datatypes.Graph]:
    """Load the QM9 dataset."""

    raw_mols_path = os.path.join(root_dir, "gdb9.sdf")
    supplier = Chem.SDMolSupplier(raw_mols_path, removeHs=False, sanitize=False)

    properties_csv_path = os.path.join(root_dir, "gdb9.sdf.csv")
    properties = pd.read_csv(properties_csv_path)
    properties.set_index("mol_id", inplace=True)

    for index, mol in enumerate(tqdm.tqdm(supplier, desc="Loading QM9")):
        if max_num_molecules is not None and index >= max_num_molecules:
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
                species=QM9Dataset.atomic_numbers_to_species(atomic_numbers)
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
    gdb9_txt_excluded = os.path.join(root_dir, "uncharacterized.txt")
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

    # Now, create a list of included indices.
    Ngdb9 = 133885
    included_idxs = np.array(sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))
    return included_idxs, excluded_idxs
    
def get_qm9_splits(
    root_dir: str,
    edm_splits: bool,
) -> Dict[str, np.ndarray]:
    """Adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9/data/prepare/qm9.py."""
    included_idxs, excluded_idxs = remove_uncharacterized_molecules(root_dir)

    # Now, generate random permutations to assign molecules to training/valation/test sets.
    Nmols = len(included_idxs)
    Ntrain = 100000
    Ntest = int(0.1 * Nmols)
    Nval = Nmols - (Ntrain + Ntest)

    # Generate random permutation.
    np.random.seed(0)
    if edm_splits:
        data_permutation = np.random.permutation(Nmols)
    else:
        data_permutation = np.arange(Nmols)

    train, val, test, extra = np.split(
        data_permutation, [Ntrain, Ntrain + Nval, Ntrain + Nval + Ntest]
    )

    assert len(extra) == 0, "Split was inexact {} {} {} {}".format(
        len(train), len(val), len(test), len(extra)
    )

    train = included_idxs[train]
    val = included_idxs[val]
    test = included_idxs[test]

    splits = {"train": train, "val": val, "test": test}

    # Cleanup file.
    try:
        os.remove(gdb9_txt_excluded)
    except OSError:
        pass

    return splits
