from typing import Dict, Iterable, Optional, List, Tuple
import os
import logging
import json
import zipfile

import numpy as np

from atomic_datasets import datatypes
from atomic_datasets import utils


# Zenodo URL for preprocessed data
GEOM_DRUGS_ZENODO_URL = "https://zenodo.org/record/18484634/files/geom_drugs_processed.zip"


class GEOMDrugs(datatypes.MolecularDataset):
    """
    The GEOM (Drugs) dataset from https://www.nature.com/articles/s41597-022-01288-4.
    
    Loads preprocessed data from Zenodo (memory-mapped for efficiency).

    Args:
        root_dir: Directory to store/load data
        split: Which split to use ('train', 'val', 'test')
        start_index: Start index for slicing the dataset
        end_index: End index for slicing the dataset
        max_atoms: Filter out molecules with more atoms than this
        conformer_selection: How to select conformers ('first', 'random', 'all')
        random_seed: Random seed for conformer selection (if conformer_selection='random')
        mmap_mode: Memory-map mode for numpy arrays ('r', 'r+', 'c', or None to load into memory)
    
    Example:
        >>> dataset = GEOMDrugs(root_dir="data/geom", split="train")
        >>> print(len(dataset))
        >>> graph = dataset[0]
        >>> print(graph.properties["smiles"])
    """

    # Atomic numbers present in GEOM-Drugs
    ATOMIC_NUMBERS = np.asarray([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83])

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        max_atoms: Optional[int] = None,
        conformer_selection: str = "all",
        random_seed: int = 0,
        mmap_mode: Optional[str] = 'r',
    ):
        super().__init__()
        self.root_dir = os.path.join(root_dir, "geom_drugs")
        self.split = split
        self.start_index = start_index
        self.end_index = end_index
        self.max_atoms = max_atoms
        self.conformer_selection = conformer_selection
        self.random_seed = random_seed
        self.mmap_mode = mmap_mode

        self.preprocessed = False
        
        # Data storage (initialized in preprocess)
        self._positions = None          # (N_total, 3) memory-mapped
        self._species = None            # (N_total,) memory-mapped
        self._atom_types = None         # (N_total,) memory-mapped (indices)
        self._atom_type_lookup = None   # Index to symbol mapping
        self._offsets = None            # (n_conformers + 1,) start indices
        self._n_atoms = None            # (n_conformers,) atoms per conformer
        self._mol_indices = None        # (n_conformers,) molecule index
        self._smiles = None             # List of SMILES strings
        self._indices = None            # Indices into conformers (after filtering)
        self._rng = None                # Random number generator

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        
        if conformer_selection not in ("first", "random", "all"):
            raise ValueError(f"conformer_selection must be 'first', 'random', or 'all', got '{conformer_selection}'")

    @classmethod
    def atom_types(cls) -> np.ndarray:
        return utils.atomic_numbers_to_symbols(cls.get_atomic_numbers())

    @classmethod
    def get_atomic_numbers(cls) -> np.ndarray:
        return cls.ATOMIC_NUMBERS
    
    @classmethod
    def atomic_numbers_to_species(cls, atomic_numbers: np.ndarray) -> np.ndarray:
        """Map atomic numbers to species indices."""
        return np.searchsorted(cls.ATOMIC_NUMBERS, atomic_numbers)

    def preprocess(self):
        """Initialize data access - downloads if needed, then loads."""
        if self.preprocessed:
            return
        self.preprocessed = True

        # Download and extract if needed
        self._ensure_downloaded()
        
        # Load data
        self._load_data()
        
        # Setup indices based on conformer selection
        self._rng = np.random.default_rng(self.random_seed)
        self._setup_indices()
    
    def _ensure_downloaded(self):
        """Download and extract preprocessed files from Zenodo if not present."""
        os.makedirs(self.root_dir, exist_ok=True)
        
        # Check if data is already extracted by looking for a key file
        marker_file = os.path.join(self.root_dir, "train_positions.npy")
        if os.path.exists(marker_file):
            return
        
        # Download zip file
        zip_filename = "geom_drugs_processed.zip"
        zip_path = os.path.join(self.root_dir, zip_filename)
        
        if not os.path.exists(zip_path):
            print(f"Downloading {zip_filename}...")
            utils.download_url(GEOM_DRUGS_ZENODO_URL, self.root_dir, filename=zip_filename)
        
        # Extract zip file
        print(f"Extracting {zip_filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.root_dir)
        
        # Optionally remove the zip file to save space
        os.remove(zip_path)
        print("Extraction complete.")
    
    def _load_data(self):
        """Load preprocessed data from numpy files."""
        prefix = self.split
        
        print(f"Loading GEOM-Drugs {self.split} split from {self.root_dir}")
        
        # Load memory-mapped arrays
        self._positions = np.load(
            os.path.join(self.root_dir, f"{prefix}_positions.npy"),
            mmap_mode=self.mmap_mode
        )
        self._species = np.load(
            os.path.join(self.root_dir, f"{prefix}_species.npy"),
            mmap_mode=self.mmap_mode
        )
        self._atom_types = np.load(
            os.path.join(self.root_dir, f"{prefix}_atom_types.npy"),
            mmap_mode=self.mmap_mode
        )
        
        # Load regular arrays (small, no need for mmap)
        self._offsets = np.load(os.path.join(self.root_dir, f"{prefix}_offsets.npy"))
        self._n_atoms = np.load(os.path.join(self.root_dir, f"{prefix}_n_atoms.npy"))
        self._mol_indices = np.load(os.path.join(self.root_dir, f"{prefix}_mol_indices.npy"))
        self._atom_type_lookup = np.load(
            os.path.join(self.root_dir, f"{prefix}_atom_type_lookup.npy"),
            allow_pickle=True
        )
        
        # Load SMILES from JSON
        with open(os.path.join(self.root_dir, f"{prefix}_smiles.json")) as f:
            self._smiles = json.load(f)
        
        n_molecules = len(np.unique(self._mol_indices))
        n_conformers = len(self._n_atoms)
        print(f"Loaded {n_molecules} molecules with {n_conformers} total conformers")
    
    def _setup_indices(self):
        """Setup indices based on conformer selection mode."""
        n_conformers = len(self._n_atoms)
        
        if self.conformer_selection == "all":
            self._indices = np.arange(n_conformers)
        else:
            # Find first index of each molecule
            unique_mols, first_indices = np.unique(self._mol_indices, return_index=True)
            
            if self.conformer_selection == "first":
                self._indices = first_indices
            elif self.conformer_selection == "random":
                counts = np.diff(np.append(first_indices, n_conformers))
                offsets = np.array([self._rng.integers(0, c) for c in counts])
                self._indices = first_indices + offsets
        
        # Apply max_atoms filter
        if self.max_atoms is not None:
            mask = self._n_atoms[self._indices] <= self.max_atoms
            n_filtered = len(self._indices) - mask.sum()
            if n_filtered > 0:
                logging.info(f"Filtered out {n_filtered} conformers with > {self.max_atoms} atoms")
            self._indices = self._indices[mask]
        
        # Apply start/end index
        if self.start_index is not None:
            self._indices = self._indices[self.start_index:]
        if self.end_index is not None:
            self._indices = self._indices[:self.end_index]

    @utils.after_preprocess
    def __len__(self) -> int:
        return len(self._indices)

    @utils.after_preprocess
    def __getitem__(self, idx: int) -> datatypes.Graph:
        """Fast random access to a conformer."""
        if idx < 0:
            idx = len(self._indices) + idx
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._indices)}")
        
        real_idx = self._indices[idx]
        
        # Slice using offsets
        start = self._offsets[real_idx]
        end = self._offsets[real_idx + 1]
        
        # Extract data for this conformer
        positions = np.array(self._positions[start:end])
        species = np.array(self._species[start:end])
        atom_type_indices = self._atom_types[start:end]
        atom_types = self._atom_type_lookup[atom_type_indices]
        
        return datatypes.Graph(
            nodes=dict(
                positions=positions,
                species=species,
                atom_types=atom_types,
            ),
            edges=None,
            senders=None,
            receivers=None,
            n_edge=None,
            n_node=np.asarray([self._n_atoms[real_idx]]),
            globals=None,
            properties=dict(smiles=self._smiles[real_idx]),
        )

    @utils.after_preprocess
    def __iter__(self) -> Iterable[datatypes.Graph]:
        """Iterate over all conformers."""
        for i in range(len(self)):
            yield self[i]
    
    @utils.after_preprocess
    def get_num_conformers(self, mol_idx: int) -> int:
        """Get number of conformers for a molecule by its molecule index."""
        return np.sum(self._mol_indices == mol_idx)
    
    @utils.after_preprocess
    def get_molecule_indices(self) -> np.ndarray:
        """Get array mapping each conformer to its molecule index."""
        return self._mol_indices[self._indices]