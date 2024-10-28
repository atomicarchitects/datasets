from typing import List, Iterable, Dict

import os
import urllib

import logging
import numpy as np
import ase
import jraph
import rdkit.Chem as Chem

from atomic_datasets import utils
from atomic_datasets import InMemoryDataset

QM9_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
)


class QM9Dataset(InMemoryDataset):
    """QM9 dataset."""

    def __init__(
        self,
        root_dir: str,
        check_molecule_sanity: bool,
        use_edm_splits: bool,
        num_train_molecules: int,
        num_val_molecules: int,
        num_test_molecules: int,
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.check_molecule_sanity = check_molecule_sanity
        self.use_edm_splits = use_edm_splits
        self.num_train_molecules = num_train_molecules
        self.num_val_molecules = num_val_molecules
        self.num_test_molecules = num_test_molecules
        self.all_structures = None

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1, 6, 7, 8, 9])

    def structures(self) -> Iterable[jraph.GraphsTuple]:
        if self.all_structures is None:
            self.all_structures = load_qm9(self.root_dir, self.check_molecule_sanity)

        return self.all_structures

    def split_indices(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of indices for each split."""
        splits = get_qm9_splits(self.root_dir, edm_splits=self.use_edm_splits)
        requested_splits = {
            "train": self.num_train_molecules,
            "val": self.num_val_molecules,
            "test": self.num_test_molecules,
        }
        for split_name, num_molecules in requested_splits.items():
            original_split_size = len(splits[split_name])
            if num_molecules > original_split_size:
                raise ValueError(
                    f"Requested {num_molecules} molecules for split {split_name}, but only {original_split_size} are available."
                )
            logging.info(
                f"Using {num_molecules} molecules out of {original_split_size} in split {split_name}.",
            )
            splits[split_name] = splits[split_name][:num_molecules]
        return splits


def molecule_sanity(mol: Chem.Mol) -> bool:
    """Check that the molecule passes some basic sanity checks from Posebusters.
    Source: https://github.com/maabuu/posebusters/blob/main/posebusters/modules/sanity.py
    """

    errors = Chem.rdmolops.DetectChemistryProblems(
        mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_ALL
    )
    types = [error.GetType() for error in errors]
    num_frags = len(Chem.rdmolops.GetMolFrags(mol, asMols=False, sanitizeFrags=False))

    results = {
        "passes_valence_checks": "AtomValenceException" not in types,
        "passes_kekulization": "AtomKekulizeException" not in types,
        "passes_rdkit_sanity_checks": len(errors) == 0,
        "all_atoms_connected": num_frags <= 1,
    }
    return all(results.values())


def load_qm9(
    root_dir: str,
    check_molecule_sanity: bool = True,
) -> List[ase.Atoms]:
    """Load the QM9 dataset."""

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    path = utils.download_url(QM9_URL, root_dir)
    utils.extract_zip(path, root_dir)

    raw_mols_path = os.path.join(root_dir, "gdb9.sdf")
    supplier = Chem.SDMolSupplier(raw_mols_path, removeHs=False, sanitize=False)

    atomic_numbers = QM9Dataset.get_atomic_numbers()
    all_structures = []
    for mol in supplier:
        if mol is None:
            continue

        # Check that the molecule passes some basic checks from Posebusters.
        if check_molecule_sanity:
            sane = molecule_sanity(mol)
            if not sane:
                continue

        # Convert to Structure.
        structure = jraph.GraphsTuple(
            nodes=dict(
                positions=np.asarray(mol.GetConformer().GetPositions()),
                species=np.searchsorted(
                    atomic_numbers,
                    np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
                ),
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=None,
            n_node=np.asarray([mol.GetNumAtoms()]),
            n_edge=None,
        )
        all_structures.append(structure)

    return all_structures


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
