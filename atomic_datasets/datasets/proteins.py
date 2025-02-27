from typing import List, Iterable, Dict, Optional

import os
import logging
import warnings

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb

from atomic_datasets import datatypes
from atomic_datasets import utils


CATH_URL = "http://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S20-v4_3_0.pdb.tgz"
MINIPROTEIN_URL = "https://files.ipd.uw.edu/pub/robust_de_novo_design_minibinders_2021/supplemental_files/scaffolds.tar.gz"


class Proteins(datatypes.MolecularDataset):
    """Generic protein structure dataset."""

    def __init__(
        self,
        root_dir: str,
        dataset: str,
        split: str,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        train_on_single_molecule: Optional[bool] = False,
        train_on_single_molecule_index: Optional[int] = 0,
        alpha_carbons_only: Optional[bool] = False,
        rng_seed: Optional[int] = 6489,  # Taken from FoldingDiff: https://github.com/microsoft/foldingdiff
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.split = split
        self.train_on_single_molecule = train_on_single_molecule

        if self.train_on_single_molecule:
            logging.info(
                f"Training on a single molecule with index {train_on_single_molecule_index}."
            )
            self.start_index = train_on_single_molecule_index
            self.end_index = train_on_single_molecule_index + 1
        else:
            self.start_index = start_index
            self.end_index = end_index

        self.all_graphs = None
        self.rng = np.random.default_rng(seed=rng_seed)
        self.dataset = dataset
        self.alpha_carbons_only = alpha_carbons_only

    @staticmethod
    def get_atomic_numbers(alpha_carbons_only: bool) -> np.ndarray:
        return (
            np.asarray([6]) if alpha_carbons_only else np.asarray([6, 7])
        )  # representing residues by their CB atoms

    @staticmethod
    def species_to_atomic_numbers(alpha_carbons_only: bool) -> Dict[int, int]:
        if alpha_carbons_only:
            return {0: 6}
        mapping = {}
        # C first, then CA, then amino acids
        for i in range(24):
            mapping[i] = 6
        mapping[24] = 7  # N
        mapping[25] = 7  # X
        return mapping

    @staticmethod
    def atoms_to_species(alpha_carbons_only: bool) -> Dict[str, int]:
        if alpha_carbons_only:
            return {"CA": 0}
        mapping = {}
        amino_acid_abbr = Proteins.get_amino_acids()
        for i, aa in enumerate(amino_acid_abbr):
            mapping[aa] = i
        mapping["C"] = 22
        mapping["CA"] = 23
        mapping["N"] = 24
        mapping["X"] = 25
        return mapping

    def num_species(self) -> int:
        return len(Proteins.get_atomic_numbers(self.alpha_carbons_only))

    @staticmethod
    def get_amino_acids() -> List[str]:
        return [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "PYL",
            "SEC",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ]

    @staticmethod
    def get_species(alpha_carbons_only) -> List[str]:
        if alpha_carbons_only:
            return ["CA"]
        return Proteins.get_amino_acids() + ["C", "CA", "N"]
    
    def preprocess(self):
        if self.all_graphs is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.all_graphs = load_data(
                    self.dataset,
                    self.root_dir,
                    self.alpha_carbons_only,
                )
        splits = self.split_indices()
        split = splits[self.split]
        if self.start_index is not None:
            split = split[self.start_index :]
        if self.end_index is not None:
            split = split[: self.end_index]
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

    def split_indices(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of indices for each split."""
        if self.train_on_single_molecule:
            return {
                "train": [self.train_on_single_molecule_index],
                "val": [self.train_on_single_molecule_index],
                "test": [self.train_on_single_molecule_index],
            }

        # using cath splits from foldingdiff
        indices = np.arange(len(self.all_structures))
        self.rng.shuffle(indices)
        splits = {
            "train": np.arange(self.num_train_molecules),
            "val": np.arange(
                self.num_train_molecules,
                self.num_train_molecules + self.num_val_molecules,
            ),
            "test": np.arange(
                self.num_train_molecules + self.num_val_molecules,
                min(
                    len(self.all_structures),
                    (
                        self.num_train_molecules
                        + self.num_val_molecules
                        + self.num_test_molecules
                    ),
                ),
            ),
        }
        splits = {k: indices[v] for k, v in splits.items()}
        return splits


class CATH(Proteins):
    """CATH protein structure dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        train_on_single_molecule: Optional[bool] = False,
        train_on_single_molecule_index: Optional[int] = 0,
        alpha_carbons_only: Optional[bool] = False,
        rng_seed: Optional[int] = 6489,  # Taken from FoldingDiff: https://github.com/microsoft/foldingdiff
    ):
        super().__init__(
            root_dir,
            "cath",
            split,
            start_index,
            end_index,
            train_on_single_molecule,
            train_on_single_molecule_index,
            alpha_carbons_only,
            rng_seed,
        )
    

class Miniproteins(Proteins):
    """Miniproteins protein structure dataset."""

    def __init__(
        self,
        root_dir: str,
        split: str,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        train_on_single_molecule: Optional[bool] = False,
        train_on_single_molecule_index: Optional[int] = 0,
        alpha_carbons_only: Optional[bool] = False,
        rng_seed: Optional[int] = 6489,  # Taken from FoldingDiff: https://github.com/microsoft/foldingdiff
    ):
        super().__init__(
            root_dir,
            "miniprotein",
            split,
            start_index,
            end_index,
            train_on_single_molecule,
            train_on_single_molecule_index,
            alpha_carbons_only,
            rng_seed,
        )


def load_data(
    dataset: str,
    root_dir: str,
    alpha_carbons_only: bool = False,
) -> List[datatypes.Graph]:
    """Load the dataset."""

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if dataset == "cath":
        mols_path = os.path.join(root_dir, "dompdb")
        if not os.path.isdir(mols_path):
            logging.info("Downloading cath dataset...")
            path = utils.download_url(CATH_URL, root_dir)
            utils.extract_tar(path, root_dir)
        mol_files_list = os.listdir(mols_path)
    elif dataset == "miniprotein":
        mols_path = os.path.join(root_dir, "supplemental_files", "scaffolds")
        scaffolds_path = os.path.join(mols_path, "recommended_scaffolds.list")
        if not os.path.isfile(scaffolds_path):
            logging.info("Downloading miniprotein dataset...")
            path = utils.download_url(MINIPROTEIN_URL, root_dir)
            utils.extract_tar(path, root_dir)
        with open(scaffolds_path, "r") as scaffolds_file:
            mol_files_list = scaffolds_file.readlines()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    all_structures = []

    def _add_structure(pos, spec, molfile, residue_starts):
        assert len(pos) == len(spec), (
            f"Length mismatch: {len(pos)} vs {len(spec)} in {molfile}"
        )

        pos = np.asarray(pos)
        spec = np.asarray(spec)

        # Convert to Structure.
        structure = datatypes.Graph(
            nodes=dict(
                positions=pos,
                species=spec,
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=dict(
                num_residues=np.asarray([len(residue_starts)]),
                residue_starts=residue_starts,
                n_short_edge=None,
                n_long_edge=None,
            ),
            n_node=np.asarray([len(spec)]),
            n_edge=None,
        )
        all_structures.append(structure)

    logging.info("Loading structures...")
    for mol_file in mol_files_list:
        mol_path = os.path.join(mols_path, mol_file).strip()
        # read pdb
        f = pdb.PDBFile.read(mol_path)
        structure = pdb.get_structure(f)
        backbone = structure.get_array(0)
        if alpha_carbons_only:
            mask = np.isin(backbone.atom_name, ["CA"])
        else:
            mask = np.isin(backbone.atom_name, ["CA", "N", "C", "CB"])
        backbone = backbone[mask]
        max_len = 4.0 if alpha_carbons_only else 2.6
        fragment_starts = np.concatenate(
            [
                np.array([0]),
                # distance between CB and N is ~2.4 angstroms + some wiggle room
                struc.check_backbone_continuity(backbone, max_len=max_len),
                np.array([len(backbone)]),
            ]
        )
        for i in range(len(fragment_starts) - 1):
            fragment = backbone[fragment_starts[i] : fragment_starts[i + 1]]
            try:
                positions = fragment.coord
                elements = fragment.atom_name
                if not alpha_carbons_only:
                    first_n = np.argwhere(elements == "N")[0][0]
                    elements[first_n] = "X"
                    # set CB to corresponding residue name
                    cb_atoms = np.argwhere(fragment.atom_name == "CB").flatten()
                    elements[cb_atoms] = fragment.res_name[cb_atoms]
                species = np.vectorize(Proteins.atoms_to_species(alpha_carbons_only).get)(
                    elements
                )
                residue_starts = struc.get_residue_starts(fragment)
                _add_structure(positions, species, mol_file, residue_starts)
            except Exception as e:
                print(e)
                continue

    logging.info(f"Loaded {len(all_structures)} structures.")
    return all_structures
