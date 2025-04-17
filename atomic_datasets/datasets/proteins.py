from typing import Generator, List, Iterable, Dict, Optional

import os
import pickle
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


amino_acid_dict = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",
    "SER": "S",
    "THR": "T",
    "SEC": "U",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


class ProteinsGeneric(datatypes.MolecularDataset):
    """Generic protein structure dataset."""

    def __init__(
        self,
        root_dir: str,
        dataset: str,
        split: Optional[str] = None,
        use_random_splits: bool = True,
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
        self.use_random_splits = use_random_splits
        self.start_index = start_index
        self.end_index = end_index
        self.split = split
        self.train_on_single_molecule = train_on_single_molecule
        self.preprocessed = False
        self.max_residues = max_residues

        if self.use_random_splits:
            if self.split is None:
                raise ValueError(
                    "When use_random_splits is True, split must be provided."
                )

            if self.start_index is not None or self.end_index is not None:
                logging.warning(
                    "When use_random_splits is True, start_index and end_index refer to the indices of the random splits."
                )

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.all_graphs is None:
                self.all_graphs = list(self.load_data())

        if not self.use_random_splits:
            return

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
    
    def load_data(self):
        return load_data(
            self.dataset,
            self.root_dir,
            ProteinsAlphaCarbons.atoms_to_species(),
            mode="alpha_carbons",
            use_random_splits=self.use_random_splits,
            max_residues=self.max_residues,
            start_index=self.start_index,
            end_index=self.end_index,
        )

    @classmethod
    def atom_types(cls) -> np.ndarray:
        return cls.get_species()

    @classmethod
    def get_atomic_numbers(cls) -> np.ndarray:
        return np.asarray([6])

    @classmethod
    def species_to_atomic_numbers(cls) -> Dict[int, int]:
        return {0: 6}

    @classmethod
    def atoms_to_species(cls) -> Dict[str, int]:
        return {"CA": 0}

    @classmethod
    def get_species(cls) -> List[str]:
        return ["CA"]


class ProteinsBackbone(ProteinsGeneric):
    
    def load_data(self):
        return load_data(
            self.dataset,
            self.root_dir,
            ProteinsBackbone.atoms_to_species(),
            mode="backbone",
            use_random_splits=self.use_random_splits,
            max_residues=self.max_residues,
            start_index=self.start_index,
            end_index=self.end_index,
        )

    @classmethod
    def atom_types(cls) -> np.ndarray:
        return cls.get_species()

    @classmethod
    def get_atomic_numbers(cls) -> np.ndarray:
        return np.asarray([0] * 22 + [6, 7, 7])

    @classmethod
    def species_to_atomic_numbers(cls) -> Dict[int, int]:
        mapping = {}
        # C first, then CA, then amino acids
        for i in range(23):
            mapping[i] = 6
        mapping[23] = 7  # N
        mapping[24] = 7  # X = initial N
        return mapping

    @classmethod
    def atoms_to_species(cls) -> Dict[str, int]:
        mapping = {}
        amino_acid_abbr = get_amino_acids()
        for i, aa in enumerate(amino_acid_abbr):
            mapping[aa] = i
        mapping["C"] = 22
        mapping["CA"] = 23
        mapping["N"] = 24
        # mapping["X"] = 25
        return mapping

    @classmethod
    def get_species(cls) -> List[str]:
        return get_amino_acids() + ["C", "CA", "N"]


class ProteinsFull(ProteinsGeneric):
    
    def load_data(self):
        return load_data(
            self.dataset,
            self.root_dir,
            ProteinsBackbone.atoms_to_species(),
            mode="full",
            use_random_splits=self.use_random_splits,
            max_residues=self.max_residues,
            start_index=self.start_index,
            end_index=self.end_index,
        )

    @classmethod
    def atom_types(cls) -> np.ndarray:
        return cls.get_species()

    @classmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1, 6, 7, 8, 16, 34])

    @classmethod
    def species_to_atomic_numbers(cls) -> Dict[int, int]:
        return {
            0: 1,
            1: 6,
            2: 7,
            3: 8,
            4: 16,
            5: 34,
            6: 7,  # X
        }

    @classmethod
    def atoms_to_species() -> Dict[str, int]:
        mapping = {}
        mapping["H"] = 0
        mapping["C"] = 1
        mapping["N"] = 2
        mapping["O"] = 3
        mapping["S"] = 4
        mapping["Se"] = 5
        # mapping["X"] = 6
        return mapping

    @classmethod
    def get_species() -> List[str]:
        return ["H", "C", "N", "O", "S", "Se"]


class SplitterMixin:
    """Retrieves dataset splits."""

    def __init__(
        self,
        num_train_molecules: int,
        num_val_molecules: int,
        num_test_molecules: int,
        train_on_single_molecule: bool = False,
        train_on_single_molecule_index: Optional[int] = 0,
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
        total_mols = (
            self.num_train_molecules + self.num_val_molecules + self.num_test_molecules
        )
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
    """Dataset of CATH proteins."""

    def __init__(
        self,
        rng_seed: Optional[
            int
        ] = 6489,  # Taken from FoldingDiff: https://github.com/microsoft/foldingdiff
        **kwargs,
    ):
        super().__init__(
            num_train_molecules=16793,
            num_val_molecules=2100,
            num_test_molecules=2099,
            dataset="cath",
            rng_seed=rng_seed,
            **kwargs,
        )


class MiniproteinsAlphaCarbons(SplitterMixin, ProteinsAlphaCarbons):
    """Dataset of miniproteins with alpha carbons only."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            num_train_molecules=53445,
            num_val_molecules=6681,
            num_test_molecules=6681,
            dataset="miniproteins",
            **kwargs,
        )


class MiniproteinsBackbone(SplitterMixin, ProteinsBackbone):
    """Dataset of miniproteins with backbone atoms only."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            num_train_molecules=53445,
            num_val_molecules=6681,
            num_test_molecules=6681,
            dataset="miniproteins",
            **kwargs,
        )


class Miniproteins(SplitterMixin, ProteinsFull):
    """Dataset of miniproteins."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            num_train_molecules=53445,
            num_val_molecules=6681,
            num_test_molecules=6681,
            dataset="miniproteins",
            **kwargs,
        )


def create_structure(
    positions: np.ndarray,
    species: np.ndarray,
    elements: np.ndarray,
    aa_sequence: str,
    mol_file: str,
) -> datatypes.Graph:
    return datatypes.Graph(
        nodes=dict(
            positions=np.asarray(positions),
            species=np.asarray(species),
            atom_types=np.asarray(elements),
        ),
        edges=None,
        receivers=None,
        senders=None,
        globals=dict(
            aa_sequence=aa_sequence,
            num_residues=len(aa_sequence),
            file_name=mol_file,
        ),
        n_node=np.asarray([len(species)]),
        n_edge=None,
    )


def load_data(
    dataset: str,
    root_dir: str,
    atoms_to_species: Dict[str, int],
    mode: str,
    use_random_splits: bool,
    max_residues: Optional[int] = None,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> Iterable[datatypes.Graph]:
    """Load the dataset."""

    if use_random_splits:
        start_index = None
        end_index = None
    
    pickle_file = os.path.join(
        root_dir, f"{dataset}_{mode}_maxlength={max_residues}_start={start_index}_end={end_index}.pkl"
    )
    if os.path.isfile(pickle_file):
        logging.info(f"Loading preprocessed {dataset} dataset.")
        with open(pickle_file, "rb") as f:
            all_structures = pickle.load(f)
        logging.info(f"Loaded {len(all_structures)} structures.")
        return all_structures

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

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

    if max_residues is None:
        max_residues = np.inf

    logging.info("Loading structures...")
    count = 0
    for index, mol_file in enumerate(
        sorted(mol_files_list),
    ):
        if start_index is not None and index < start_index:
            continue

        if end_index is not None and index >= end_index:
            break

        mol_path = os.path.join(mols_path, mol_file).strip()
        # print(f"Processing {mol_path}...")
        # read pdb
        f = pdb.PDBFile.read(mol_path)
        try:
            structure = pdb.get_structure(f)
        except:
            print(f"Could not read {mol_path}")
            continue

        # filter out non-backbone atoms, if necessary
        protein = structure.get_array(0)
        if mode == "alpha_carbons":
            mask = np.isin(protein.atom_name, ["CA"])
        elif mode == "backbone":
            mask = np.isin(protein.atom_name, ["CA", "N", "C", "CB"])
        elif mode == "full":
            mask = np.ones(len(protein), dtype=bool)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        protein = protein[mask]

        # get starts of protein chains
        max_len = 5.0 if mode == "alpha_carbons" else 1.8  # (default 1.8)
        chain_starts = np.concatenate(
            [
                np.array([0]),
                struc.check_backbone_continuity(protein, max_len=max_len),
                np.array([len(protein)]),
            ]
        )

        # get processed structures
        for i in range(len(chain_starts) - 1):
            chain = protein[chain_starts[i] : chain_starts[i + 1]]
            try:
                positions = chain.coord
                if mode == "full":
                    elements = chain.element
                else:
                    elements = chain.atom_name

                # Get initial N of the chain
                # if mode != "alpha_carbons":
                #     first_n = np.argwhere(elements == "N")[0][0]
                #     elements[first_n] = "X"

                # Set CB to corresponding residue name ("backbone" only)
                if mode == "backbone":
                    cb_atoms = np.argwhere(chain.atom_name == "CB").flatten()
                    elements[cb_atoms] = chain.res_name[cb_atoms]

                species = np.vectorize(atoms_to_species.get)(elements)

                # cut down # of residues if necessary
                residue_starts = struc.get_residue_starts(chain)
                residue_starts = np.concatenate(
                    [residue_starts, np.array([len(chain)])]
                )
                end_ndx = len(residue_starts) - max_residues
                if end_ndx >= 1:
                    start_residue = np.random.default_rng().integers(end_ndx)

                    if mode == "alpha_carbons":
                        start = start_residue
                        end = start_residue + max_residues
                    else:
                        start = residue_starts[start_residue]
                        end = residue_starts[start_residue + max_residues]

                    positions = positions[start:end]
                    species = species[start:end]
                    elements = elements[start:end]

                assert len(positions) >= 5, f"Too few atoms in {mol_file}"

                aa_sequence = np.vectorize(amino_acid_dict.get)(
                    chain.res_name[residue_starts[:-1]]
                )
                structure = create_structure(
                    positions,
                    species,
                    elements,
                    aa_sequence,
                    mol_file,
                )
                all_structures.append(structure)
                count += 1

            except Exception as e:
                print(f"Error processing {mol_file}: {e}")
                print(f"Skipping {mol_file}...")
                continue

    logging.info(f"Loaded {count} structures.")
    with open(pickle_file, "wb") as f:
        pickle.dump(all_structures, f)
    return all_structures
