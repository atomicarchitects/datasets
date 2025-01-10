from typing import Iterable, Dict

import os
import urllib

import logging
import numpy as np
import pandas as pd
import rdkit.Chem as Chem

from atomic_datasets import utils
from atomic_datasets import datatypes

QM9_URL = r"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"


class QM9Dataset(datatypes.MolecularDataset):
    """QM9 dataset."""

    def __init__(
        self,
        root_dir: str,
        check_with_rdkit: bool,
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.check_with_rdkit = check_with_rdkit
        self.preprocessed = False

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1, 6, 7, 8, 9])

    def preprocess(self):
        preprocess(self.root_dir)
        self.preprocessed = True

    def __iter__(self) -> Iterable[datatypes.Graph]:
        if not self.preprocessed:
            self.preprocess()

        while True:
            yield from load_qm9(self.root_dir, self.check_with_rdkit)


def preprocess(root_dir: str):
    """Preprocess the files for the QM9 dataset."""
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
) -> Iterable[datatypes.Graph]:
    """Load the QM9 dataset."""

    raw_mols_path = os.path.join(root_dir, "gdb9.sdf")
    supplier = Chem.SDMolSupplier(raw_mols_path, removeHs=False, sanitize=False)

    properties_csv_path = os.path.join(root_dir, "gdb9.sdf.csv")
    properties = pd.read_csv(properties_csv_path)

    for mol in supplier:
        if mol is None:
            raise ValueError("Failed to load molecule.")

        # Check that the molecule passes some basic checks from Posebusters.
        if check_with_rdkit and not utils.is_molecule_sane(mol):
            continue

        mol_id = mol.GetProp("_Name")
        mol_properties = properties[properties["mol_id"] == mol_id].to_dict(orient="records")[0]

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


def get_qm9_splits(
    root_dir: str,
    edm_splits: bool,
) -> Dict[str, np.ndarray]:
    """Adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9/data/prepare/qm9.py."""

    def is_int(string: str) -> bool:
        try:
            int(string)
            return True
        except:
            return False

    logging.info("Dropping uncharacterized molecules.")
    gdb9_url_excluded = "https://springernature.figshare.com/ndownloader/files/3195404"
    gdb9_txt_excluded = os.path.join(root_dir, "uncharacterized.txt")
    urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)

    # First, get list of excluded indices.
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0] for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert (
        len(excluded_idxs) == 3054
    ), "There should be exactly 3054 excluded atoms. Found {}".format(
        len(excluded_idxs)
    )

    # Now, create a list of included indices.
    Ngdb9 = 133885
    Nexcluded = 3054

    included_idxs = np.array(sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

    # Now, generate random permutations to assign molecules to training/valation/test sets.
    Nmols = Ngdb9 - Nexcluded
    assert Nmols == len(
        included_idxs
    ), "Number of included molecules should be equal to Ngdb9 - Nexcluded. Found {} {}".format(
        Nmols, len(included_idxs)
    )

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
