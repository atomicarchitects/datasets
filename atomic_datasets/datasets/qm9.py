from typing import Iterable, Dict, Optional, Tuple, Union
import os
import logging

import numpy as np
import pandas as pd
import tqdm

from atomic_datasets import utils
from atomic_datasets import datatypes

QM9_URL = (
    r"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
)


class QM9(datatypes.MolecularDataset):
    """
    The QM9 dataset from https://www.nature.com/articles/sdata201422.
    
    Contains ~134k small organic molecules with up to 9 heavy atoms (C, N, O, F).
    
    Args:
        root_dir: Directory to store/load data
        split: Which split to use ('train', 'val', 'test') if use_Anderson_splits=True
        use_Anderson_splits: Use the Anderson et al. (https://arxiv.org/abs/1906.04015) Cormorant paper splits (100k train, ~13k val, ~13k test)
        check_with_rdkit: Validate molecules with RDKit sanity checks
        check_validity: Check molecule validity (slower)
        start_index: Start index for slicing
        end_index: End index for slicing
    
    Example:
        >>> dataset = QM9(root_dir="data/qm9", use_Anderson_splits=True, split="train")
        >>> print(len(dataset))
        >>> graph = dataset[0]  # Fast random access
    """

    # Atomic numbers present in QM9: H, C, N, O, F
    ATOMIC_NUMBERS = np.asarray([1, 6, 7, 8, 9])

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        use_Anderson_splits: bool = False,
        check_with_rdkit: bool = False,
        check_validity: bool = False,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.split = split
        self.use_Anderson_splits = use_Anderson_splits
        self.check_with_rdkit = check_with_rdkit
        self.check_validity = check_validity
        self.start_index = start_index
        self.end_index = end_index
        
        self.preprocessed = False
        
        # Data storage
        self._positions = None      # List of (N, 3) arrays
        self._species = None        # List of (N,) arrays  
        self._atom_types = None     # List of (N,) arrays
        self._n_atoms = None        # Array of molecule sizes
        self._indices = None        # Indices after filtering/splitting
        self._properties = None     # List of property dicts

        if self.use_Anderson_splits and self.split is None:
            raise ValueError("When use_Anderson_splits is True, split must be provided.")

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
        """Load and preprocess QM9 data."""
        if self.preprocessed:
            return
        self.preprocessed = True

        # Print README path
        readme_path = os.path.join(self.root_dir, "QM9_README")
        print("Dataset description available at:", os.path.abspath(readme_path))

        # Check for cached preprocessed data
        cache_file = os.path.join(self.root_dir, "qm9_preprocessed.npz")
        
        if os.path.exists(cache_file) and not self.check_with_rdkit and not self.check_validity:
            self._load_from_cache(cache_file)
        else:
            self._load_from_raw()
            if not self.check_with_rdkit and not self.check_validity:
                self._save_to_cache(cache_file)
        
        # Apply splits
        n_molecules = len(self._positions)
        
        if self.use_Anderson_splits:
            splits = _get_edm_splits(self.root_dir, n_molecules)
            self._indices = splits[self.split].copy()
        else:
            self._indices = np.arange(n_molecules)
        
        # Apply start/end index
        if self.start_index is not None:
            self._indices = self._indices[self.start_index:]
        if self.end_index is not None:
            self._indices = self._indices[:self.end_index]
    
    def _load_from_cache(self, cache_file: str):
        """Load preprocessed data from cache."""
        print(f"Loading QM9 from cache: {os.path.abspath(cache_file)}")
        data = np.load(cache_file, allow_pickle=True)
        
        self._positions = list(data['positions'])
        self._species = list(data['species'])
        self._atom_types = list(data['atom_types'])
        self._n_atoms = data['n_atoms']
        self._properties = list(data['properties'])
        
        print(f"Loaded {len(self._positions)} molecules from cache")
    
    def _save_to_cache(self, cache_file: str):
        """Save preprocessed data to cache."""
        print(f"Saving QM9 cache to: {cache_file}")
        np.savez(
            cache_file,
            positions=np.array(self._positions, dtype=object),
            species=np.array(self._species, dtype=object),
            atom_types=np.array(self._atom_types, dtype=object),
            n_atoms=self._n_atoms,
            properties=np.array(self._properties, dtype=object),
        )
    
    def _load_from_raw(self):
        """Load from raw SDF file."""
        import rdkit.Chem as Chem
        
        # Download if needed
        _download_if_needed(self.root_dir)
        
        raw_mols_path = os.path.join(self.root_dir, "gdb9.sdf")
        supplier = Chem.SDMolSupplier(raw_mols_path, removeHs=False, sanitize=False)
        
        # Load properties from CSV
        properties_csv_path = os.path.join(self.root_dir, "gdb9.sdf.csv")
        properties_df = pd.read_csv(properties_csv_path)
        properties_df.set_index("mol_id", inplace=True)
        
        self._positions = []
        self._species = []
        self._atom_types = []
        self._properties = []
        n_atoms_list = []
        
        for index, mol in enumerate(tqdm.tqdm(supplier, desc="Loading QM9")):
            if mol is None:
                logging.warning(f"Failed to load molecule {index}")
                continue
            
            # Optional checks
            if self.check_with_rdkit and not utils.is_molecule_sane(mol):
                logging.info(f"Skipping molecule {index} due to sanity check failure")
                continue
            if self.check_validity and not utils.check_molecule_validity(mol):
                logging.info(f"Skipping molecule {index} due to validity check failure")
                continue
            
            # Extract data
            positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
            atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            species = self.atomic_numbers_to_species(atomic_numbers)
            atom_types = utils.atomic_numbers_to_symbols(atomic_numbers)
            
            # Extract properties from CSV
            mol_id = mol.GetProp("_Name")
            mol_properties = properties_df.loc[mol_id].to_dict()
            mol_properties["mol_id"] = mol_id
            mol_properties["smiles"] = Chem.MolToSmiles(mol)
            
            self._positions.append(positions)
            self._species.append(species)
            self._atom_types.append(atom_types)
            self._properties.append(mol_properties)
            n_atoms_list.append(len(positions))
        
        self._n_atoms = np.array(n_atoms_list, dtype=np.int32)
        print(f"Loaded {len(self._positions)} molecules")

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
            properties=self._properties[real_idx],
        )

    @utils.after_preprocess
    def __iter__(self) -> Iterable[datatypes.Graph]:
        """Iterate over all molecules."""
        for i in range(len(self)):
            yield self[i]


# =============================================================================
# Helper functions
# =============================================================================

def _download_if_needed(root_dir: str):
    """Download and extract QM9 if not already present."""
    raw_mols_path = os.path.join(root_dir, "gdb9.sdf")
    
    if os.path.exists(raw_mols_path):
        return
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    print(f"Downloading QM9 dataset to {root_dir}")
    path = utils.download_url(QM9_URL, root_dir)
    utils.extract_zip(path, root_dir)
    print("Download complete.")


def _get_edm_splits(root_dir: str, n_molecules: int) -> Dict[str, np.ndarray]:
    """
    Get train/val/test splits following EDM paper.
    
    Removes uncharacterized molecules and splits:
    - Train: 100,000 molecules
    - Val: ~13,000 molecules  
    - Test: ~13,000 molecules
    """
    included_idxs = _get_included_indices(root_dir, n_molecules)
    
    n_included = len(included_idxs)
    n_train = 100000
    n_test = int(0.1 * n_included)
    n_val = n_included - n_train - n_test
    
    # Deterministic shuffle
    np.random.seed(0)
    perm = np.random.permutation(n_included)
    
    train_perm = perm[:n_train]
    val_perm = perm[n_train:n_train + n_val]
    test_perm = perm[n_train + n_val:]
    
    return {
        'train': included_idxs[train_perm],
        'val': included_idxs[val_perm],
        'test': included_idxs[test_perm],
    }


def _get_included_indices(root_dir: str, n_molecules: int) -> np.ndarray:
    """Get indices of molecules that are not in the excluded list."""
    
    # Download excluded molecules list
    excluded_url = "https://springernature.figshare.com/ndownloader/files/3195404"
    excluded_file = os.path.join(root_dir, "uncharacterized.txt")
    
    if not os.path.exists(excluded_file):
        utils.download_url(excluded_url, root_dir)
        # Rename downloaded file
        downloaded = os.path.join(root_dir, "3195404")
        if os.path.exists(downloaded):
            os.rename(downloaded, excluded_file)
    
    # Parse excluded indices
    excluded_idxs = set()
    if os.path.exists(excluded_file):
        with open(excluded_file) as f:
            for line in f:
                parts = line.split()
                if parts and parts[0].isdigit():
                    excluded_idxs.add(int(parts[0]) - 1)  # Convert to 0-indexed
    
    # Create included indices
    included = np.array([i for i in range(n_molecules) if i not in excluded_idxs])
    
    print(f"QM9: {len(included)} included, {len(excluded_idxs)} excluded")
    return included
