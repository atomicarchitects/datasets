import os
import json
import zipfile
from typing import Dict, Iterable, Optional, Sequence, Any

import numpy as np

from atomic_datasets import datatypes
from atomic_datasets import utils


# Zenodo URL for preprocessed data
ChEMBL3D_ZENODO_URL = "https://zenodo.org/records/18488050/files/chembl3d_processed.zip"


class ChEMBL3D(datatypes.MolecularDataset):
    """
    Dataset of ChEMBL3D structures from https://github.com/isayevlab/LoQI.
    
    Contains structures for a large subset of ChEMBL, pre-optimized or extracted
    from 3D experimental data. This implementation uses a high-performance 
    contiguous memory layout with memory-mapping support.

    Args:
        root_dir: Directory to store/load data
        split: Which split to use ('train', 'val', 'test_small', 'test_rot_bonds', 'test_cremp')
        start_index: Start index for slicing the dataset
        end_index: End index for slicing the dataset
        train_on_single_molecule: If True, use single molecule for all splits
        train_on_single_molecule_index: Index of molecule to use if train_on_single_molecule=True
        mmap_mode: Memory-map mode for numpy arrays ('r', 'r+', 'c', or None to load into memory)
    """

    ATOMIC_NUMBERS = np.asarray([
        1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83, 34
    ], dtype=np.int32)

    def __init__(
        self,
        root_dir: str,
        split: str,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        train_on_single_molecule: bool = False,
        train_on_single_molecule_index: int = 0,
        mmap_mode: Optional[str] = 'r',
    ):
        super().__init__(atomic_numbers=self.ATOMIC_NUMBERS)

        self.root_dir = os.path.join(root_dir, "chembl3d")
        self.split = split
        self.start_index = start_index
        self.end_index = end_index
        self.train_on_single_molecule = train_on_single_molecule
        self.train_on_single_molecule_index = train_on_single_molecule_index
        self.mmap_mode = mmap_mode

        self.preprocessed = False
        
        # Data storage - will be initialized as contiguous arrays or memory-mapped views
        self._positions = None      # (total_atoms, 3) float32
        self._species = None        # (total_atoms,) uint8
        self._offsets = None        # (n_molecules + 1,) int64
        self._charges = None        # (total_atoms,) int8
        self._is_aromatic = None    # (total_atoms,) uint8
        self._is_in_ring = None     # (total_atoms,) uint8
        self._hybridization = None  # (total_atoms,) uint8
        self._smiles = None         # list of str
        self._chemblids = None      # list of str
        self._indices = None        # indices after filtering/slicing

        self.preprocess()

    def _get_file_path(self, filename: str) -> str:
        """Helper to get the absolute path to a preprocessed file."""
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
        
        # Clean up zip file to save disk space
        os.remove(zip_path)
        print("Download complete.")

    def preprocess(self):
        """Load preprocessed numpy files and setup indices."""
        if self.preprocessed:
            return

        prefix = self.split
        
        # Check that files exist, download if needed
        positions_path = self._get_file_path(f"{prefix}_positions.npy")
        if not os.path.exists(positions_path):
            self._download_from_zenodo()
        
        print(f"Loading ChEMBL3D {self.split} split from: {self.root_dir}")
        
        # Load arrays (memory-mapped for large files to save RAM)
        self._positions = np.load(self._get_file_path(f"{prefix}_positions.npy"), mmap_mode=self.mmap_mode)
        self._species = np.load(self._get_file_path(f"{prefix}_species.npy"), mmap_mode=self.mmap_mode)
        self._offsets = np.load(self._get_file_path(f"{prefix}_offsets.npy"))  # Small, load fully into RAM
        self._charges = np.load(self._get_file_path(f"{prefix}_charges.npy"), mmap_mode=self.mmap_mode)
        self._is_aromatic = np.load(self._get_file_path(f"{prefix}_is_aromatic.npy"), mmap_mode=self.mmap_mode)
        self._is_in_ring = np.load(self._get_file_path(f"{prefix}_is_in_ring.npy"), mmap_mode=self.mmap_mode)
        self._hybridization = np.load(self._get_file_path(f"{prefix}_hybridization.npy"), mmap_mode=self.mmap_mode)
        
        # Load metadata (SMILES and IDs)
        metadata_path = self._get_file_path(f"{prefix}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self._smiles = metadata["smiles"]
        self._chemblids = metadata["chemblids"]
        
        n_molecules = len(self._offsets) - 1
        n_atoms = len(self._positions)
        print(f"Loaded {n_molecules:,} molecules, {n_atoms:,} total atoms")
        
        # Apply single molecule override for debugging or overfitting tests
        if self.train_on_single_molecule:
            self._indices = np.array([self.train_on_single_molecule_index])
        else:
            self._indices = np.arange(n_molecules)
        
        # Apply start/end index for subsetting
        self._indices = self._indices[slice(self.start_index, self.end_index)]
        
        self.preprocessed = True

    def __len__(self) -> int:
        """Returns the number of molecules in the current split/slice."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> datatypes.Graph:
        """
        Fast random access to a molecule using offset-based slicing.
        
        Returns:
            A datatypes.Graph object containing positions, species, and atomic properties.
        """
        if idx < 0:
            idx = len(self._indices) + idx
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._indices)}")
        
        mol_idx = self._indices[idx]
        start = self._offsets[mol_idx]
        end = self._offsets[mol_idx + 1]
        
        # Slice arrays (copies from mmap into RAM for the return object)
        species = np.array(self._species[start:end])
        atomic_numbers = self.species_to_atomic_numbers(species)
        atom_types = utils.atomic_numbers_to_symbols(atomic_numbers)
        
        return datatypes.Graph(
            nodes=dict(
                positions=np.array(self._positions[start:end]),
                atomic_numbers=atomic_numbers,
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