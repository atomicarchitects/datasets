from typing import Iterable, Dict, Optional, Tuple

import os
import logging
import json
import zipfile

import numpy as np

from atomic_datasets import utils
from atomic_datasets import datatypes

# Zenodo URL for preprocessed data
CROSSDOCKED_ZENODO_URL = "https://zenodo.org/records/18584578/files/crossdocked.zip"


class CrossDocked(datatypes.MolecularDataset):
    """
    The CrossDocked dataset from https://pubs.acs.org/doi/full/10.1021/acs.jcim.0c00411
    with splits from Luo et al. (https://proceedings.neurips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html).

    Loads preprocessed data from Zenodo (memory-mapped for efficiency).

    Args:
        root_dir: Directory to store/load data
        split: Which split to use ('train', 'val', 'test')
        start_index: Start index for slicing the dataset
        end_index: End index for slicing the dataset
        mmap_mode: Memory-map mode for numpy arrays ('r', 'r+', 'c', or None to load into memory)
    """

    ATOMIC_NUMBERS = np.asarray(
        [
            13,
            33,
            79,
            5,
            35,
            6,
            17,
            27,
            24,
            29,
            9,
            26,
            1,
            80,
            53,
            3,
            12,
            42,
            7,
            8,
            15,
            44,
            16,
            21,
            34,
            14,
            50,
            23,
            74,
            39,
        ],
        dtype=np.int32,
    )

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        mmap_mode: Optional[str] = "r",
    ):
        super().__init__(atomic_numbers=self.ATOMIC_NUMBERS)

        self.root_dir = os.path.join(root_dir, "crossdocked")
        self.split = split
        self.start_index = start_index
        self.end_index = end_index
        self.mmap_mode = mmap_mode

        self.preprocessed = False

        # Data storage
        self._positions = None  # (N_total, 3) memory-mapped
        self._atom_types = None  # (N_total,) memory-mapped (indices into lookup)
        self._offsets = None  # (n_complexes + 1,) start indices
        self._n_atoms = None  # (n_complexes,) atoms per complex
        self._atom_type_lookup = None  # (n_types,) symbol strings
        self._properties = None  # List of dicts (pocket_file, starting_fragment_mask)
        self._indices = None  # Indices after slicing

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.preprocess()

        readme_path = os.path.join(self.root_dir, "README.md")
        if os.path.exists(readme_path):
            print("Dataset description available at:", os.path.abspath(readme_path))

    def preprocess(self):
        """Initialize data access - downloads if needed, then loads."""
        if self.preprocessed:
            return

        self._ensure_downloaded()
        self._load_data()
        self._setup_indices()

        self.preprocessed = True

    def _ensure_downloaded(self):
        """Download and extract preprocessed files from Zenodo if not present."""
        os.makedirs(self.root_dir, exist_ok=True)

        # Check if data is already extracted
        marker_file = os.path.join(self.root_dir, "crossdocked", "train_positions.npy")
        if os.path.exists(marker_file):
            return

        zip_filename = "crossdocked.zip"
        zip_path = os.path.join(self.root_dir, zip_filename)

        if not os.path.exists(zip_path):
            utils.download_url(
                CROSSDOCKED_ZENODO_URL, self.root_dir, filename=zip_filename
            )

        print(f"Extracting {zip_filename}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.root_dir)

        os.remove(zip_path)
        print("Extraction complete.")

    def _load_data(self):
        """Load preprocessed data using memory mapping."""
        prefix = self.split
        print(f"Loading CrossDocked {self.split} split from {self.root_dir}")

        self._positions = np.load(
            os.path.join(self.root_dir, "crossdocked", f"{prefix}_positions.npy"),
            mmap_mode=self.mmap_mode,
        )
        self._atom_types = np.load(
            os.path.join(self.root_dir, "crossdocked", f"{prefix}_atom_types.npy"),
            mmap_mode=self.mmap_mode,
        )

        self._offsets = np.load(
            os.path.join(self.root_dir, "crossdocked", f"{prefix}_offsets.npy")
        )
        self._n_atoms = np.load(
            os.path.join(self.root_dir, "crossdocked", f"{prefix}_n_atoms.npy")
        )
        self._atom_type_lookup = np.load(
            os.path.join(self.root_dir, "crossdocked", f"{prefix}_atom_type_lookup.npy")
        )

        with open(
            os.path.join(self.root_dir, "crossdocked", f"{prefix}_properties.json")
        ) as f:
            self._properties = json.load(f)

        n_complexes = len(self._n_atoms)
        print(f"Loaded {n_complexes} complexes")

    def _setup_indices(self):
        """Setup indices with optional slicing."""
        n_complexes = len(self._n_atoms)
        self._indices = np.arange(n_complexes)
        self._indices = self._indices[slice(self.start_index, self.end_index)]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> datatypes.Graph:
        """Fast slice access via memory-mapped offsets."""
        if idx < 0:
            idx = len(self._indices) + idx

        real_idx = self._indices[idx]
        start, end = self._offsets[real_idx], self._offsets[real_idx + 1]

        species = np.array(self._atom_types[start:end])
        atom_types = self._atom_type_lookup[species]
        atomic_numbers = utils.atomic_symbols_to_numbers(atom_types)

        return datatypes.Graph(
            nodes=dict(
                positions=np.array(self._positions[start:end]),
                atomic_numbers=atomic_numbers,
                species=species,
                atom_types=atom_types,
            ),
            edges=None,
            senders=None,
            receivers=None,
            n_edge=None,
            n_node=np.asarray([self._n_atoms[real_idx]]),
            globals=None,
            properties=self._properties[real_idx],
        )
