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


def get_amino_acids():
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


class ProteinsGeneric(datatypes.MolecularDataset):
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
        max_residues: Optional[int] = None,
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.split = split
        self.train_on_single_molecule = train_on_single_molecule
        self.preprocessed = False
        self.max_residues = max_residues

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
        self.dataset = dataset
    
    def load_data(self):
        raise NotImplementedError

    def preprocess(self):
        self.preprocessed = True
        if self.all_graphs is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.all_graphs = self.load_data()
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
        raise NotImplementedError

class ProteinsAlphaCarbons(ProteinsGeneric):
    def __init__(
        self,
        root_dir: str,
        dataset: str,
        split: str,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        train_on_single_molecule: Optional[bool] = False,
        train_on_single_molecule_index: Optional[int] = 0,
        max_residues: Optional[int] = None,
    ):
        super().__init__(
            root_dir,
            dataset,
            split,
            start_index,
            end_index,
            train_on_single_molecule,
            train_on_single_molecule_index,
            max_residues,
        )
    
    def load_data(self):
        return load_data(
            self.dataset,
            self.root_dir,
            ProteinsAlphaCarbons.atoms_to_species(),
            alpha_carbons_only=True,
            max_residues=self.max_residues,
        )
    
    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([6])
    
    @staticmethod
    def species_to_atomic_numbers() -> Dict[int, int]:
        return {0: 6}
    
    @staticmethod
    def atoms_to_species() -> Dict[str, int]:
        return {"CA": 0}

    @staticmethod
    def get_species() -> List[str]:
        return ["CA"]

class ProteinsBackbone(ProteinsGeneric):
    def __init__(
        self,
        root_dir: str,
        dataset: str,
        split: str,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        train_on_single_molecule: Optional[bool] = False,
        train_on_single_molecule_index: Optional[int] = 0,
        max_residues: Optional[int] = None,
    ):
        super().__init__(
            root_dir,
            dataset,
            split,
            start_index,
            end_index,
            train_on_single_molecule,
            train_on_single_molecule_index,
            max_residues,
        )
    
    def load_data(self):
        return load_data(
            self.dataset,
            self.root_dir,
            ProteinsBackbone.atoms_to_species(),
            alpha_carbons_only=False,
            max_residues=self.max_residues,
        )
    
    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([6, 7])  # representing residues by their CB atoms

    @classmethod
    def species_to_atomic_numbers(cls, species: np.ndarray) -> Dict[int, int]:
        mapping = {}
        # C first, then CA, then amino acids
        for i in range(24):
            mapping[i] = 6
        mapping[24] = 7  # N
        mapping[25] = 7  # X
        return np.vectorize(mapping.get)(species)

    @staticmethod
    def atoms_to_species() -> Dict[str, int]:
        mapping = {}
        amino_acid_abbr = get_amino_acids()
        for i, aa in enumerate(amino_acid_abbr):
            mapping[aa] = i
        mapping["C"] = 22
        mapping["CA"] = 23
        mapping["N"] = 24
        mapping["X"] = 25
        return mapping

    @staticmethod
    def get_species() -> List[str]:
        return get_amino_acids() + ["C", "CA", "N"]

class SplitterMixin:
    """Retrieves dataset splits."""

    def __init__(
        self,
        num_train_molecules: int,
        num_val_molecules: int,
        num_test_molecules: int,
        train_on_single_molecule: bool,
        train_on_single_molecule_index: int,
        rng_seed: Optional[int] = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_train_molecules = num_train_molecules
        self.num_val_molecules = num_val_molecules
        self.num_test_molecules = num_test_molecules
        self.train_on_single_molecule = train_on_single_molecule
        self.train_on_single_molecule_index = train_on_single_molecule_index
        self.rng = np.random.default_rng(seed=rng_seed)

    def split_indices(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of indices for each split."""
        if self.train_on_single_molecule:
            return {
                "train": [self.train_on_single_molecule_index],
                "val": [self.train_on_single_molecule_index],
                "test": [self.train_on_single_molecule_index],
            }

        # using cath splits from foldingdiff
        total_mols = self.num_train_molecules + self.num_val_molecules + self.num_test_molecules
        indices = np.arange(total_mols)
        self.rng.shuffle(indices)
        splits = {
            "train": np.arange(self.num_train_molecules),
            "val": np.arange(
                self.num_train_molecules,
                self.num_train_molecules + self.num_val_molecules,
            ),
            "test": np.arange(
                self.num_train_molecules + self.num_val_molecules,
                min(len(self.all_graphs), total_mols),
            ),
        }
        splits = {k: indices[v] for k, v in splits.items()}
        return splits


class CATHAlphaCarbons(SplitterMixin, ProteinsAlphaCarbons):
    def __init__(
        self,
        root_dir: str,
        split: str,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        train_on_single_molecule: Optional[bool] = False,
        train_on_single_molecule_index: Optional[int] = 0,
        max_residues: Optional[int] = None,
        rng_seed: Optional[int] = 6489,  # Taken from FoldingDiff: https://github.com/microsoft/foldingdiff
    ):
        super().__init__(
            16793,
            2100,
            2099,
            train_on_single_molecule,
            train_on_single_molecule_index,
            rng_seed,
            root_dir=root_dir,
            dataset="cath",
            split=split,
            start_index=start_index,
            end_index=end_index,
            max_residues=max_residues,
        )


class MiniproteinsAlphaCarbons(SplitterMixin, ProteinsAlphaCarbons):
    def __init__(
        self,
        root_dir: str,
        split: str,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        train_on_single_molecule: Optional[bool] = False,
        train_on_single_molecule_index: Optional[int] = 0,
        max_residues: Optional[int] = None,
        rng_seed: Optional[int] = 0,
    ):
        super().__init__(
            53446,
            6681,
            6681,
            train_on_single_molecule,
            train_on_single_molecule_index,
            rng_seed,
            root_dir=root_dir,
            dataset="miniproteins",
            split=split,
            start_index=start_index,
            end_index=end_index,
            max_residues=max_residues,
        )

def load_data(
    dataset: str,
    root_dir: str,
    atoms_to_species: Dict[str, int],
    alpha_carbons_only: bool,
    max_residues: Optional[int] = None,
) -> List[datatypes.Graph]:
    """Load the dataset."""

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if max_residues is None:
        max_residues = np.inf

    # Download raw data.
    if dataset == "cath":
        mols_path = os.path.join(root_dir, "dompdb")
        if not os.path.isdir(mols_path):
            logging.info("Downloading cath dataset...")
            path = utils.download_url(CATH_URL, root_dir)
            utils.extract_tar(path, root_dir)
        mol_files_list = os.listdir(mols_path)
    elif dataset == "miniproteins":
        mols_path = os.path.join(root_dir, "supplemental_files", "scaffolds")
        scaffolds_path = os.path.join(mols_path, "all_scaffolds.list")
        if not os.path.isfile(scaffolds_path):
            logging.info("Downloading miniproteins dataset...")
            path = utils.download_url(MINIPROTEIN_URL, root_dir)
            utils.extract_tar(path, root_dir)
        with open(scaffolds_path, "r") as scaffolds_file:
            mol_files_list = scaffolds_file.readlines()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    all_structures = []

    def _add_structure(pos, spec, molfile):
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
            globals=None,
            n_node=np.asarray([len(spec)]),
            n_edge=None,
        )
        all_structures.append(structure)

    logging.info("Loading structures...")
    for mol_file in mol_files_list:
        mol_path = os.path.join(mols_path, mol_file).strip()
        # read pdb
        f = pdb.PDBFile.read(mol_path)
        try:
            structure = pdb.get_structure(f)
        except:
            print(f"Could not read {mol_path}")
            continue
        backbone = structure.get_array(0)
        if alpha_carbons_only:
            mask = np.isin(backbone.atom_name, ["CA"])
        else:
            mask = np.isin(backbone.atom_name, ["CA", "N", "C", "CB"])
        backbone = backbone[mask]
        max_len = 5.0 if alpha_carbons_only else 3.0
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
                species = np.vectorize(atoms_to_species.get)(
                    elements
                )
                # cut down # of residues if necessary
                residue_starts = struc.get_residue_starts(fragment)
                residue_starts = np.concatenate([residue_starts, np.array([len(fragment)])])
                if len(residue_starts) - 1 > max_residues:
                    start = np.random.randint(0, len(residue_starts) - max_residues)
                    residue_starts = residue_starts[start : start + max_residues + 1]
                    start = residue_starts[start] if not alpha_carbons_only else start
                    end = start + max_residues if alpha_carbons_only else residue_starts[-1]
                    positions = positions[start : end]
                    species = species[start : end]
                assert len(positions) >= 5, f"Too few atoms in {mol_file}"
                _add_structure(positions, species, mol_file)
            except Exception as e:
                # print(e)
                continue

    logging.info(f"Loaded {len(all_structures)} structures.")
    return all_structures
