from typing import Iterable, Dict, Optional
import os
import logging

import tqdm
import numpy as np
import ase.io

from atomic_datasets import utils
from atomic_datasets import datatypes


TMQM_URL = r"https://github.com/bbskjelstad/tmqm.git"


class tmQM(datatypes.MolecularDataset):
    """
    The tmQM dataset from https://pubs.acs.org/doi/10.1021/acs.jcim.0c01041.
    
    Contains ~86k transition metal complexes with elements from H (1) to Hg (80).
    
    Args:
        root_dir: Directory to store/load data
        split: Which split to use ('train', 'val', 'test')
        use_default_splits: Use default splits (69k train, 9k val, ~8.7k test)
        splits: Custom split definitions (ignored if use_default_splits=True)
        start_index: Start index for slicing
        end_index: End index for slicing
        rng_seed: Random seed for shuffling
        train_on_single_molecule: If True, use single molecule for all splits
        train_on_single_molecule_index: Index of molecule to use if train_on_single_molecule=True
    
    Example:
        >>> dataset = tmQM(root_dir="data/tmqm", split="train")
        >>> print(len(dataset))
        >>> graph = dataset[0]  # Fast random access
    """

    # Atomic numbers present in tmQM: H (1) to Hg (80)
    ATOMIC_NUMBERS = np.arange(1, 81)

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        use_default_splits: bool = True,
        splits: Optional[Dict[str, np.ndarray]] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        rng_seed: int = 0,
        train_on_single_molecule: bool = False,
        train_on_single_molecule_index: int = 0,
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.split = split
        self.start_index = start_index
        self.end_index = end_index
        self.rng_seed = rng_seed
        self.train_on_single_molecule = train_on_single_molecule
        self.train_on_single_molecule_index = train_on_single_molecule_index

        if use_default_splits:
            if splits is not None:
                raise ValueError("If use_default_splits is True, splits must be None.")
            self.splits = {
                "train": np.arange(69000),
                "val": np.arange(69000, 78000),
                "test": np.arange(78000, 86665),
            }
        else:
            if splits is None:
                raise ValueError(
                    "If use_default_splits is False, splits must be provided."
                )
            self.splits = splits

        self.preprocessed = False
        
        # Data storage (like QM9)
        self._positions = None      # List of (N, 3) arrays
        self._species = None        # List of (N,) arrays  
        self._atom_types = None     # List of (N,) arrays
        self._n_atoms = None        # Array of molecule sizes
        self._indices = None        # Indices after filtering/splitting

    @classmethod
    def atom_types(cls) -> np.ndarray:
        return utils.atomic_numbers_to_symbols(cls.get_atomic_numbers())

    @classmethod
    def get_atomic_numbers(cls) -> np.ndarray:
        return cls.ATOMIC_NUMBERS

    @classmethod
    def species_to_atomic_numbers(cls) -> Dict[int, int]:
        return {i: i + 1 for i in range(80)}
    
    @classmethod
    def atomic_numbers_to_species(cls, atomic_numbers: np.ndarray) -> np.ndarray:
        """Map atomic numbers to species indices (0-indexed)."""
        return atomic_numbers - 1

    def preprocess(self):
        """Load and preprocess tmQM data."""
        if self.preprocessed:
            return
        self.preprocessed = True

        # Check for cached preprocessed data
        cache_file = os.path.join(self.root_dir, "tmqm_preprocessed.npz")
        
        if os.path.exists(cache_file):
            self._load_from_cache(cache_file)
        else:
            self._load_from_raw()
            self._save_to_cache(cache_file)
        
        # Apply splits
        n_molecules = len(self._positions)
        split_indices = self._get_split_indices(n_molecules)
        
        if self.split is not None:
            self._indices = split_indices[self.split].copy()
        else:
            self._indices = np.arange(n_molecules)
        
        # Apply start/end index
        if self.start_index is not None:
            self._indices = self._indices[self.start_index:]
        if self.end_index is not None:
            self._indices = self._indices[:self.end_index]

    def _load_from_cache(self, cache_file: str):
        """Load preprocessed data from cache."""
        print(f"Loading tmQM from cache: {os.path.abspath(cache_file)}")
        data = np.load(cache_file, allow_pickle=True)
        
        self._positions = list(data['positions'])
        self._species = list(data['species'])
        self._atom_types = list(data['atom_types'])
        self._n_atoms = data['n_atoms']
        
        print(f"Loaded {len(self._positions)} molecules from cache")
    
    def _save_to_cache(self, cache_file: str):
        """Save preprocessed data to cache."""
        print(f"Saving tmQM cache to: {cache_file}")
        np.savez(
            cache_file,
            positions=np.array(self._positions, dtype=object),
            species=np.array(self._species, dtype=object),
            atom_types=np.array(self._atom_types, dtype=object),
            n_atoms=self._n_atoms,
        )

    def _load_from_raw(self):
        """Load from raw XYZ files."""
        xyzs_path = _download_data(self.root_dir)
        
        self._positions = []
        self._species = []
        self._atom_types = []
        n_atoms_list = []
        
        xyz_files = sorted(os.listdir(xyzs_path))
        
        for mol_file in tqdm.tqdm(xyz_files, desc="Loading tmQM"):
            mol_path = os.path.join(xyzs_path, mol_file)
            
            try:
                mol_as_ase = ase.io.read(mol_path, format="xyz")
            except Exception as e:
                logging.warning(f"Failed to load {mol_file}: {e}")
                continue
                
            if mol_as_ase is None:
                continue

            positions = np.asarray(mol_as_ase.positions, dtype=np.float32)
            atomic_numbers = mol_as_ase.numbers
            species = self.atomic_numbers_to_species(atomic_numbers)
            atom_types = utils.atomic_numbers_to_symbols(atomic_numbers)
            
            self._positions.append(positions)
            self._species.append(species)
            self._atom_types.append(atom_types)
            n_atoms_list.append(len(positions))
        
        self._n_atoms = np.array(n_atoms_list, dtype=np.int32)
        print(f"Loaded {len(self._positions)} molecules")

    def _get_split_indices(self, n_molecules: int) -> Dict[str, np.ndarray]:
        """Return a dictionary of indices for each split."""
        if self.train_on_single_molecule:
            return {
                "train": np.array([self.train_on_single_molecule_index]),
                "val": np.array([self.train_on_single_molecule_index]),
                "test": np.array([self.train_on_single_molecule_index]),
            }

        # Shuffle indices deterministically
        rng = np.random.default_rng(self.rng_seed)
        indices = np.arange(n_molecules)
        rng.shuffle(indices)
        
        # Map split ranges to shuffled indices
        splits = {k: indices[v[v < n_molecules]] for k, v in self.splits.items()}
        return splits

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
        
        real_idx = self._indices[idx]
        
        return datatypes.Graph(
            nodes=dict(
                positions=self._positions[real_idx],
                species=self._species[real_idx],
                atom_types=self._atom_types[real_idx],
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=None,
            n_node=np.asarray([self._n_atoms[real_idx]]),
            n_edge=None,
        )

    @utils.after_preprocess
    def __iter__(self) -> Iterable[datatypes.Graph]:
        """Iterate over all molecules."""
        for i in range(len(self)):
            yield self[i]


def _download_data(root_dir: str) -> str:
    """Download and extract the tmQM dataset."""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    xyzs_path = os.path.join(root_dir, "xyz")
    if os.path.exists(xyzs_path) and len(os.listdir(xyzs_path)) > 0:
        print(f"Using downloaded data: {xyzs_path}")
        return xyzs_path

    print(f"Cloning tmQM repository to {root_dir}")
    _ = utils.clone_url(TMQM_URL, root_dir)
    os.makedirs(xyzs_path)

    for i in range(1, 3):
        gz_path = os.path.join(root_dir, "tmqm/tmQM", f"tmQM_X{i}.xyz.gz")

        mol_file = utils.extract_gz(gz_path)
        with open(mol_file, "r") as f:
            all_xyzs = f.read().split("\n\n")
            for xyz_n, xyz in enumerate(all_xyzs):
                if xyz == "":
                    continue
                xyz_lines = xyz.split("\n")
                assert len(xyz_lines) == int(xyz_lines[0]) + 2
                with open(os.path.join(xyzs_path, f"X{i}_{xyz_n}.xyz"), "w") as f:
                    f.write(xyz)

    return xyzs_path
