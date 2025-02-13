from typing import List, Iterable, Dict

from absl import logging
import os
import numpy as np
import ase
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import warnings

from symphony.data import datasets
from symphony import datatypes


CATH_URL = "http://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S20-v4_3_0.pdb.tgz"
MINIPROTEIN_URL = "https://files.ipd.uw.edu/pub/robust_de_novo_design_minibinders_2021/supplemental_files/scaffolds.tar.gz"


class ProteinDataset(datasets.InMemoryDataset):
    """CATH/Miniprotein datasets."""

    def __init__(
        self,
        root_dir: str,
        num_train_molecules: int,
        num_val_molecules: int,
        num_test_molecules: int,
        dataset: str,
        train_on_single_molecule: bool = False,
        train_on_single_molecule_index: int = 0,
        alpha_carbons_only: bool = False,
        rng_seed: int = 6489,  # consistent w foldingdiff
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.train_on_single_molecule = train_on_single_molecule

        if self.train_on_single_molecule:
            logging.info(
                f"Training on a single molecule with index {train_on_single_molecule_index}."
            )
            self.num_train_molecules = 1
            self.num_val_molecules = 1
            self.num_test_molecules = 1
        else:
            self.num_train_molecules = num_train_molecules
            self.num_val_molecules = num_val_molecules
            self.num_test_molecules = num_test_molecules

        self.all_structures = None
        self.rng = np.random.default_rng(seed=rng_seed)
        self.dataset = dataset
        self.alpha_carbons_only = alpha_carbons_only

    @staticmethod
    def get_atomic_numbers(alpha_carbons_only: bool) -> np.ndarray:
        return np.asarray([6]) if alpha_carbons_only else np.asarray([6, 7])  # representing residues by their CB atoms

    @staticmethod
    def species_to_atomic_numbers(alpha_carbons_only: bool) -> Dict[int, int]:
        if alpha_carbons_only: return {0: 6}
        mapping = {}
        # C first, then CA, then amino acids
        for i in range(24):
            mapping[i] = 6
        mapping[24] = 7  # N
        mapping[25] = 7  # X
        return mapping
    
    @staticmethod
    def atoms_to_species(alpha_carbons_only: bool) -> Dict[str, int]:
        if alpha_carbons_only: return {"CA": 0}
        mapping = {}
        amino_acid_abbr = ProteinDataset.get_amino_acids()
        for i, aa in enumerate(amino_acid_abbr):
            mapping[aa] = i
        mapping["C"] = 22
        mapping["CA"] = 23
        mapping["N"] = 24
        mapping["X"] = 25
        return mapping

    def num_species(self) -> int:
        return len(ProteinDataset.get_atomic_numbers(self.alpha_carbons_only))

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
        if alpha_carbons_only: return ["CA"]
        return ProteinDataset.get_amino_acids() + ["C", "CA", "N"]

    def structures(self) -> Iterable[datatypes.Structures]:
        if self.all_structures is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.all_structures = load_data(
                    self.dataset,
                    self.root_dir,
                    self.alpha_carbons_only,
                )

        return self.all_structures

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
                    (self.num_train_molecules
                        + self.num_val_molecules
                        + self.num_test_molecules),
                )
            ),
        }
        splits = {k: indices[v] for k, v in splits.items()}
        return splits


def load_data(
    dataset: str,
    root_dir: str,
    alpha_carbons_only: bool = False,
) -> List[datatypes.Structures]:
    """Load the dataset."""

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if dataset == "cath":
        mols_path = os.path.join(root_dir, "dompdb")
        if not os.path.isdir(mols_path):
            logging.info("Downloading cath dataset...")
            path = datasets.utils.download_url(CATH_URL, root_dir)
            datasets.utils.extract_tar(path, root_dir)
        mol_files_list = os.listdir(mols_path)
    elif dataset == "miniprotein":
        mols_path = os.path.join(root_dir, "supplemental_files", "scaffolds")
        scaffolds_path = os.path.join(mols_path, "recommended_scaffolds.list")
        if not os.path.isfile(scaffolds_path):
            logging.info("Downloading miniprotein dataset...")
            path = datasets.utils.download_url(MINIPROTEIN_URL, root_dir)
            datasets.utils.extract_tar(path, root_dir)
        with open(scaffolds_path, "r") as scaffolds_file:
            mol_files_list = scaffolds_file.readlines()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    all_structures = []

    def _add_structure(pos, spec, molfile, residue_starts):
        assert len(pos) == len(spec), f"Length mismatch: {len(pos)} vs {len(spec)} in {molfile}"

        pos = np.asarray(pos)
        spec = np.asarray(spec)

        # Convert to Structure.
        structure = datatypes.Structures(
            nodes=datatypes.NodesInfo(
                positions=pos,
                species=spec,
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=datatypes.GlobalsInfo(
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
        fragment_starts = np.concatenate([
            np.array([0]),
            # distance between CB and N is ~2.4 angstroms + some wiggle room
            struc.check_backbone_continuity(backbone, max_len=max_len),
            np.array([len(backbone)]),
        ])
        for i in range(len(fragment_starts) - 1):
            fragment = backbone[fragment_starts[i]:fragment_starts[i + 1]]
            try:
                positions = fragment.coord
                elements = fragment.atom_name
                if not alpha_carbons_only:
                    first_n = np.argwhere(elements == "N")[0][0]
                    elements[first_n] = "X"
                    # set CB to corresponding residue name
                    cb_atoms = np.argwhere(fragment.atom_name == "CB").flatten()
                    elements[cb_atoms] = fragment.res_name[cb_atoms]
                species = np.vectorize(
                    ProteinDataset.atoms_to_species(alpha_carbons_only).get
                )(elements)
                residue_starts = struc.get_residue_starts(fragment)
                _add_structure(positions, species, mol_file, residue_starts)
            except Exception as e:
                print(e)
                continue

        
    logging.info(f"Loaded {len(all_structures)} structures.")
    return all_structures
