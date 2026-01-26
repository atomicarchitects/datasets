from typing import Dict, Iterable, Optional, Union
import os
import logging

import numpy as np
import tqdm

from atomic_datasets import datatypes
from atomic_datasets import utils


GEOM_DRUGS_URL = r"https://zenodo.org/records/7881981/files/EDM.tar.gz"


class GEOMDrugs(datatypes.MolecularDataset):
    """
    The GEOM (Drugs) dataset from https://www.nature.com/articles/s41597-022-01288-4.
    
    Supports fast random access via memory-mapped arrays and precomputed boundaries.
    
    Args:
        root_dir: Directory to store/load data
        use_GCDM_splits: Use the predefined train/val/test splits
        split: Which split to use ('train', 'val', 'test') if use_GCDM_splits=True
        start_index: Start index for slicing the dataset
        end_index: End index for slicing the dataset
        max_atoms: Filter out molecules with more atoms than this
    
    Example:
        >>> dataset = GEOMDrugs(root_dir="data/geom", use_GCDM_splits=True, split="train")
        >>> print(len(dataset))
        >>> graph = dataset[0]  # Fast random access
        >>> for graph in dataset:  # Iteration
        ...     pass
    """

    # Atomic numbers present in GEOM-Drugs
    ATOMIC_NUMBERS = np.asarray([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83])

    def __init__(
        self,
        root_dir: str,
        use_GCDM_splits: bool = False,
        split: Optional[str] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        max_atoms: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.use_GCDM_splits = use_GCDM_splits
        self.split = split
        self.start_index = start_index
        self.end_index = end_index
        self.max_atoms = max_atoms

        self.preprocessed = False
        
        # Data storage (initialized in preprocess)
        self._raw_data = None          # Memory-mapped numpy array
        self._mol_boundaries = None    # Array of molecule start/end indices
        self._indices = None           # Indices into molecules (after splits/filtering)

        if self.use_GCDM_splits and self.split is None:
            raise ValueError("When use_GCDM_splits is True, split must be provided.")

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
        """Initialize data access - downloads if needed, sets up fast indexing."""
        if self.preprocessed:
            return
        self.preprocessed = True

        # Download if needed
        _download_if_needed(self.root_dir)
        
        # Compute or load molecule boundaries
        self._mol_boundaries = _get_or_compute_boundaries(self.root_dir)
        n_molecules = len(self._mol_boundaries) - 1
        
        # Setup memory-mapped data access
        conformation_file = os.path.join(
            self.root_dir, "data", "EDM", "GEOM", "GEOM_drugs_30.npy"
        )
        self._raw_data = np.load(conformation_file, mmap_mode='r')
        
        # Determine which molecule indices to use
        if self.use_GCDM_splits:
            splits = _get_GCDM_splits(self.root_dir)
            self._indices = splits[self.split].copy()
        else:
            self._indices = np.arange(n_molecules)
        
        # Apply start/end index
        if self.start_index is not None:
            self._indices = self._indices[self.start_index:]
        if self.end_index is not None:
            self._indices = self._indices[:self.end_index]
        
        # Filter by max_atoms if specified
        if self.max_atoms is not None:
            self._indices = self._filter_by_size(self._indices)
    
    def _filter_by_size(self, indices: np.ndarray) -> np.ndarray:
        """Filter indices to only include molecules with <= max_atoms."""
        mol_sizes = np.diff(self._mol_boundaries)
        mask = mol_sizes[indices] <= self.max_atoms
        filtered = indices[mask]
        
        n_filtered = len(indices) - len(filtered)
        if n_filtered > 0:
            logging.info(f"Filtered out {n_filtered} molecules with > {self.max_atoms} atoms")
        
        return filtered
    
    def _get_molecule_data(self, mol_idx: int) -> tuple:
        """Get atomic numbers and positions for a molecule by its index."""
        start = self._mol_boundaries[mol_idx]
        end = self._mol_boundaries[mol_idx + 1]
        
        # Raw data columns: [mol_id, atomic_num, x, y, z]
        datum = np.array(self._raw_data[start:end])
        atomic_numbers = datum[:, 1].astype(int)      # Column 1 = atomic number
        positions = datum[:, 2:].astype(np.float32)   # Columns 2-4 = x, y, z
        
        return atomic_numbers, positions
    
    def _make_graph(self, atomic_numbers: np.ndarray, positions: np.ndarray) -> datatypes.Graph:
        """Create a Graph object from atomic numbers and positions."""
        species = self.atomic_numbers_to_species(atomic_numbers)
        atom_types = utils.atomic_numbers_to_symbols(atomic_numbers)
        
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
            n_node=np.array([len(atomic_numbers)]),
            globals=None,
        )

    @utils.after_preprocess
    def __len__(self) -> int:
        return len(self._indices)

    @utils.after_preprocess
    def __getitem__(self, idx: int) -> datatypes.Graph:
        """Fast random access to a molecule."""
        if idx < 0:
            idx = len(self._indices) + idx
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._indices)}")
        
        mol_idx = self._indices[idx]
        atomic_numbers, positions = self._get_molecule_data(mol_idx)
        return self._make_graph(atomic_numbers, positions)

    @utils.after_preprocess
    def __iter__(self) -> Iterable[datatypes.Graph]:
        """Iterate over all molecules."""
        for i in range(len(self)):
            yield self[i]

def _download_if_needed(root_dir: str):
    """Download and extract GEOM-Drugs if not already present."""
    conformation_file = os.path.join(
        root_dir, "data", "EDM", "GEOM", "GEOM_drugs_30.npy"
    )
    
    if os.path.exists(conformation_file):
        return
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    print(f"Downloading GEOM (Drugs) dataset to {root_dir}")
    path = utils.download_url(GEOM_DRUGS_URL, root_dir)
    path = utils.extract_gz(path)
    path = utils.extract_tar(path, root_dir)
    print("Download complete.")


def _compute_boundaries(mol_ids: np.ndarray) -> np.ndarray:
    """Compute molecule boundary indices from molecule IDs."""
    split_indices = np.where(mol_ids[:-1] != mol_ids[1:])[0] + 1
    boundaries = np.concatenate([[0], split_indices, [len(mol_ids)]])
    return boundaries.astype(np.int64)


def _get_or_compute_boundaries(root_dir: str) -> np.ndarray:
    """Get molecule boundaries, computing and caching if needed."""
    boundaries_file = os.path.join(
        root_dir, "data", "EDM", "GEOM", "GEOM_drugs_boundaries.npy"
    )
    
    if os.path.exists(boundaries_file):
        return np.load(boundaries_file)
    
    print("Computing molecule boundaries (one-time operation)...")
    conformation_file = os.path.join(
        root_dir, "data", "EDM", "GEOM", "GEOM_drugs_30.npy"
    )
    
    # Use memory mapping to avoid loading full file
    raw_data = np.load(conformation_file, mmap_mode='r')
    mol_ids = np.array(raw_data[:, 0]).astype(int)  # Need to copy from mmap
    
    boundaries = _compute_boundaries(mol_ids)
    np.save(boundaries_file, boundaries)
    
    print(f"Saved boundaries to {boundaries_file}")
    print(f"  {len(boundaries) - 1} molecules")
    
    return boundaries


def _get_GCDM_splits(root_dir: str) -> Dict[str, np.ndarray]:
    """Get train/val/test splits from GCDM paper."""
    permutation_file = os.path.join(
        root_dir, "data", "EDM", "GEOM", "GEOM_permutation.npy"
    )
    permutation = np.load(permutation_file)

    num_mol = len(permutation)
    val_proportion = 0.1
    val_split = int(num_mol * val_proportion)
    test_proportion = 0.1
    test_split = val_split + int(num_mol * test_proportion)
    
    val_indices, test_indices, train_indices = np.split(
        permutation, [val_split, test_split]
    )

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
