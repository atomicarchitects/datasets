import os
import json
import zipfile
from typing import Dict, Iterable, Optional

import numpy as np

from atomic_datasets import datatypes
from atomic_datasets import utils


# Zenodo URL for preprocessed data
ChEMBL3D_ZENODO_URL = "https://zenodo.org/records/18488050/files/chembl3d_processed.zip"  # TODO: Update after upload


class ChEMBL3D(datatypes.MolecularDataset):
    """
    Dataset of ChEMBL3D structures from https://github.com/isayevlab/LoQI.
    
    Args:
        root_dir: Directory to store/load data
        split: Which split to use ('train', 'val', 'test_small', 'test_rot_bonds', 'test_cremp')
        start_index: Start index for slicing
        end_index: End index for slicing
        train_on_single_molecule: If True, use single molecule for all splits
        train_on_single_molecule_index: Index of molecule to use if train_on_single_molecule=True
    
    Example:
        >>> dataset = ChEMBL3D(root_dir="data/ChEMBL3D_stereo", split="train")
        >>> print(len(dataset))
        >>> graph = dataset[0]  # Fast random access
    """

    # Atomic numbers present in ChEMBL3D
    ATOMIC_NUMBERS = np.asarray([
        1,   # H
        5,   # B
        6,   # C
        7,   # N
        8,   # O
        9,   # F
        13,  # Al
        14,  # Si
        15,  # P
        16,  # S
        17,  # Cl
        33,  # As
        35,  # Br
        53,  # I
        80,  # Hg
        83,  # Bi
        34,  # Se
    ], dtype=np.int32)
    
    ATOM_SYMBOLS = np.array([
        'H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi', 'Se'
    ])

    def __init__(
        self,
        root_dir: str,
        split: str,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        train_on_single_molecule: bool = False,
        train_on_single_molecule_index: int = 0,
    ):
        super().__init__()

        self.root_dir = os.path.join(root_dir, "chembl3d")
        self.split = split
        self.start_index = start_index
        self.end_index = end_index
        self.train_on_single_molecule = train_on_single_molecule
        self.train_on_single_molecule_index = train_on_single_molecule_index

        self.preprocessed = False
        
        # Data storage - memory mapped for efficiency
        self._positions = None      # (total_atoms, 3) float32
        self._species = None        # (total_atoms,) uint8
        self._offsets = None        # (n_molecules + 1,) int64
        self._charges = None        # (total_atoms,) int8
        self._is_aromatic = None    # (total_atoms,) uint8
        self._is_in_ring = None     # (total_atoms,) uint8
        self._hybridization = None  # (total_atoms,) uint8
        self._smiles = None         # list of str
        self._chemblids = None      # list of str
        self._indices = None        # indices after filtering

    @classmethod
    def atom_types(cls) -> np.ndarray:
        return cls.ATOM_SYMBOLS.copy()

    @classmethod
    def get_atomic_numbers(cls) -> np.ndarray:
        return cls.ATOMIC_NUMBERS.copy()
    
    @classmethod
    def species_to_atomic_numbers(cls) -> Dict[int, int]:
        return {i: int(an) for i, an in enumerate(cls.ATOMIC_NUMBERS)}
    
    @classmethod
    def atomic_numbers_to_species(cls, atomic_numbers: np.ndarray) -> np.ndarray:
        """Map atomic numbers to species indices."""
        lookup = {int(an): i for i, an in enumerate(cls.ATOMIC_NUMBERS)}
        return np.array([lookup[int(an)] for an in atomic_numbers], dtype=np.int32)

    def _get_file_path(self, filename: str) -> str:
        """Get path to a preprocessed file."""
        return os.path.join(self.root_dir, filename)

    def _download_from_zenodo(self):
        """Download and extract preprocessed data from Zenodo."""
        os.makedirs(self.root_dir, exist_ok=True)
        
        zip_path = os.path.join(self.root_dir, "chembl3d_processed.zip")
        
        # Download
        print(f"Downloading ChEMBL3D from Zenodo...")
        print(f"  URL: {ChEMBL3D_ZENODO_URL}")
        utils.download_url(ChEMBL3D_ZENODO_URL, self.root_dir)
        
        # Extract
        print(f"Extracting to {self.root_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Extract and flatten directory structure
            for member in zf.namelist():
                # Skip directories
                if member.endswith('/'):
                    continue
                
                # Get the filename without the parent directory
                filename = os.path.basename(member)
                if not filename:
                    continue
                
                # Extract to root_dir
                source = zf.open(member)
                target_path = os.path.join(self.root_dir, filename)
                with open(target_path, 'wb') as target:
                    target.write(source.read())
                source.close()
        
        # Clean up zip file
        os.remove(zip_path)
        print("Download complete.")

    def preprocess(self):
        """Load preprocessed numpy files."""
        if self.preprocessed:
            return
        self.preprocessed = True

        prefix = self.split
        
        # Check that files exist, download if needed
        positions_path = self._get_file_path(f"{prefix}_positions.npy")
        if not os.path.exists(positions_path):
            self._download_from_zenodo()
        
        print(f"Loading ChEMBL3D {self.split} from: {self.root_dir}")
        
        # Load arrays (memory-mapped for large files)
        self._positions = np.load(self._get_file_path(f"{prefix}_positions.npy"), mmap_mode='r')
        self._species = np.load(self._get_file_path(f"{prefix}_species.npy"), mmap_mode='r')
        self._offsets = np.load(self._get_file_path(f"{prefix}_offsets.npy"))  # Small, load fully
        self._charges = np.load(self._get_file_path(f"{prefix}_charges.npy"), mmap_mode='r')
        self._is_aromatic = np.load(self._get_file_path(f"{prefix}_is_aromatic.npy"), mmap_mode='r')
        self._is_in_ring = np.load(self._get_file_path(f"{prefix}_is_in_ring.npy"), mmap_mode='r')
        self._hybridization = np.load(self._get_file_path(f"{prefix}_hybridization.npy"), mmap_mode='r')
        
        # Load metadata
        metadata_path = self._get_file_path(f"{prefix}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self._smiles = metadata["smiles"]
        self._chemblids = metadata["chemblids"]
        
        n_molecules = len(self._offsets) - 1
        n_atoms = len(self._positions)
        print(f"Loaded {n_molecules:,} molecules, {n_atoms:,} atoms")
        
        # Apply single molecule override
        if self.train_on_single_molecule:
            self._indices = np.array([self.train_on_single_molecule_index])
        else:
            self._indices = np.arange(n_molecules)
        
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
        """Fast random access to a molecule."""
        if idx < 0:
            idx = len(self._indices) + idx
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._indices)}")
        
        mol_idx = self._indices[idx]
        start = self._offsets[mol_idx]
        end = self._offsets[mol_idx + 1]
        
        # Slice arrays (copies from mmap)
        species = np.array(self._species[start:end])
        atom_types = self.ATOM_SYMBOLS[species]
        
        return datatypes.Graph(
            nodes=dict(
                positions=np.array(self._positions[start:end]),
                species=species,
                atom_types=atom_types,
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=None,
            n_node=np.asarray([end - start]),
            n_edge=None,
            properties={
                "smiles": self._smiles[mol_idx],
                "chemblid": self._chemblids[mol_idx],
                "charges": np.array(self._charges[start:end]),
                "is_aromatic": np.array(self._is_aromatic[start:end]),
                "is_in_ring": np.array(self._is_in_ring[start:end]),
                "hybridization": np.array(self._hybridization[start:end]),
            },
        )

    @utils.after_preprocess
    def __iter__(self) -> Iterable[datatypes.Graph]:
        """Iterate over all molecules."""
        for i in range(len(self)):
            yield self[i]
