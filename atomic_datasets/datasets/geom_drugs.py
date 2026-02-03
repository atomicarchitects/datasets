from typing import Dict, Iterable, Optional, List, Tuple
import os
import logging
import pickle
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
import tqdm

from atomic_datasets import datatypes
from atomic_datasets import utils


GEOM_DRUGS_RAW_BASE_URL = r"https://bits.csb.pitt.edu/files/geom_raw/"
GEOM_DRUGS_RAW_FILES = {
    "train": "train_data.pickle",
    "val": "val_data.pickle",
    "test": "test_data.pickle",
}

# Covalent radii dictionary for topology validation (values in Angstroms)
COVALENT_RADII = {
    1: 0.31,   # Hydrogen
    6: 0.76,   # Carbon  
    7: 0.71,   # Nitrogen
    8: 0.66,   # Oxygen
    9: 0.57,   # Fluorine
    15: 1.07,  # Phosphorus
    16: 1.05,  # Sulfur
    17: 1.02,  # Chlorine
    35: 1.20,  # Bromine
    53: 1.39   # Iodine
}


def _process_molecule(mol):
    """Sanitize and kekulize a molecule. Module-level for pickling."""
    try:
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None
    
    if len(Chem.GetMolFrags(mol)) > 1:
        return None
    
    return mol


def _check_topology(adjacency_matrix, numbers, coordinates, tolerance=0.4):
    """Check if bond lengths are within tolerance. Module-level for pickling."""
    adjacency_mask = (adjacency_matrix > 0).astype(int)
    
    radii = np.array([COVALENT_RADII.get(n, 1.5) for n in numbers])
    ref_dist = (radii[:, np.newaxis] + radii[np.newaxis, :]) * adjacency_mask
    
    diff = coordinates[:, :, np.newaxis, :] - coordinates[:, np.newaxis, :, :]
    data_dist = np.linalg.norm(diff, axis=-1) * adjacency_mask
    
    diffs = np.abs(data_dist - ref_dist[np.newaxis, :, :]) <= (ref_dist[np.newaxis, :, :] * tolerance)
    return diffs.all(axis=(1, 2))


def _validate_topology(mol, tolerance=0.4):
    """Validate topology. Module-level for pickling."""
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    
    conformers = mol.GetConformers()
    if not conformers:
        return False
    
    coordinates = np.array([conf.GetPositions() for conf in conformers])
    return _check_topology(adjacency_matrix, numbers, coordinates, tolerance).all()


def _process_single_molecule_entry(args):
    """Process a single (smiles, mols) entry. Module-level for pickling."""
    smiles, mols, skip_topology, topology_tolerance, atomic_numbers_array, canonicalize_atom_order = args
    
    # Validate SMILES
    reference_mol = Chem.MolFromSmiles(smiles)
    if reference_mol is None:
        return None
    
    # Sanitize and validate conformers
    valid_conformers = []
    for mol in mols:
        sanitized = _process_molecule(mol)
        if sanitized is None:
            continue
        if skip_topology or _validate_topology(sanitized, topology_tolerance):
            valid_conformers.append(sanitized)
    
    if not valid_conformers:
        return None
    
    # Extract data from each conformer
    conformer_data = []
    for mol in valid_conformers:
        conformer = mol.GetConformer(0)
        positions = np.array(conformer.GetPositions(), dtype=np.float32)
        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        species = np.searchsorted(atomic_numbers_array, atomic_numbers)

        # Apply canonical atom ordering
        if canonicalize_atom_order:
            ranks = Chem.CanonicalRankAtoms(mol)
            order = np.argsort(ranks)
            positions = positions[order]
            atomic_numbers = atomic_numbers[order]
            species = species[order]
            
            # Renumber the mol to match the new ordering
            mol = Chem.RenumberAtoms(mol, order.tolist())

        #  Save SMILES with explicit hydrogens.
        smiles = Chem.MolToSmiles(mol, allHsExplicit=True)

        conformer_data.append({
            'positions': positions,
            'atomic_numbers': atomic_numbers,
            'species': species,
            'smiles': smiles,
        })
    
    return conformer_data


class GEOMDrugs(datatypes.MolecularDataset):
    """
    The GEOM (Drugs) dataset from https://www.nature.com/articles/s41597-022-01288-4.
    Preprocessing code adapted from https://github.com/isayevlab/geom-drugs-3dgen-evaluation.

    Args:
        root_dir: Directory to store/load data
        split: Which split to use ('train', 'val', 'test')
        start_index: Start index for slicing the dataset
        end_index: End index for slicing the dataset
        max_atoms: Filter out molecules with more atoms than this
        conformer_selection: How to select conformers ('first', 'random', 'all')
        random_seed: Random seed for conformer selection (if conformer_selection='random')
        topology_tolerance: Tolerance for topology validation (default: 0.4 = 40%)
    
    Example:
        >>> dataset = GEOMDrugs(root_dir="data/geom", split="train")
        >>> print(len(dataset))
        >>> graph = dataset[0]  # Fast random access
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
        topology_tolerance: float = 0.4,
        skip_topology_validation: bool = False,
        canonicalize_atom_order: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.start_index = start_index
        self.end_index = end_index
        self.max_atoms = max_atoms
        self.conformer_selection = conformer_selection
        self.random_seed = random_seed
        self.topology_tolerance = topology_tolerance
        self.skip_topology_validation = skip_topology_validation
        self.canonicalize_atom_order = canonicalize_atom_order

        self.preprocessed = False
        
        # Data storage (initialized in preprocess)
        self._positions = None         # List of (N, 3) arrays
        self._species = None           # List of (N,) arrays
        self._atom_types = None        # List of (N,) arrays
        self._n_atoms = None           # Array of molecule sizes
        self._smiles = None            # List of SMILES strings
        self._mol_indices = None       # Which molecule each conformer belongs to
        self._indices = None           # Indices into conformers (after filtering)
        self._rng = None               # Random number generator

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
        """Initialize data access - preprocesses raw data if needed, then loads."""
        if self.preprocessed:
            return
        self.preprocessed = True

        # Check if processed cache exists
        # Use different cache file if topology validation is skipped
        cache_suffix = "_no_topo" if self.skip_topology_validation else ""
        cache_suffix += "_canon" if self.canonicalize_atom_order else ""
        cache_file = os.path.join(self.root_dir, "processed", f"{self.split}{cache_suffix}.npz")
        
        if not os.path.exists(cache_file):
            self._run_preprocessing()
        
        # Load from cache
        self._load_from_cache(cache_file)
        
        # Setup indices based on conformer selection
        self._rng = np.random.default_rng(self.random_seed)
        self._setup_indices()
    
    def _load_from_cache(self, cache_file: str):
        """Load preprocessed data from numpy cache."""
        print(f"Loading GEOM-Drugs {self.split} split from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        
        self._positions = list(data['positions'])
        self._species = list(data['species'])
        self._atom_types = list(data['atom_types'])
        self._n_atoms = data['n_atoms']
        self._smiles = list(data['smiles'])
        self._mol_indices = data['mol_indices']
        
        n_molecules = len(np.unique(self._mol_indices))
        n_conformers = len(self._positions)
        print(f"Loaded {n_molecules} molecules with {n_conformers} total conformers")
    
    def _setup_indices(self):
        """Setup indices based on conformer selection mode."""
        n_conformers = len(self._positions)
        
        if self.conformer_selection == "all":
            # Use all conformers
            self._indices = np.arange(n_conformers)
        else:
            # Find first index of each molecule (much faster than np.where in a loop)
            # mol_indices is sorted, so we can use np.unique with return_index
            unique_mols, first_indices = np.unique(self._mol_indices, return_index=True)
            
            if self.conformer_selection == "first":
                self._indices = first_indices
            elif self.conformer_selection == "random":
                # Get count of conformers per molecule
                counts = np.diff(np.append(first_indices, n_conformers))
                # Random offset within each molecule's conformers
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
    
    def _run_preprocessing(self):
        """Run the full preprocessing pipeline on raw data."""
        raw_dir = os.path.join(self.root_dir, "raw")
        processed_dir = os.path.join(self.root_dir, "processed")
        
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Download raw data if missing
        for split_name, filename in GEOM_DRUGS_RAW_FILES.items():
            raw_path = os.path.join(raw_dir, filename)
            if not os.path.exists(raw_path):
                url = GEOM_DRUGS_RAW_BASE_URL + filename
                print(f"Downloading {filename}...")
                utils.download_url(url, raw_dir)
        
        print("Running GEOM-Drugs preprocessing pipeline...")
        
        for split_name in ["train", "val", "test"]:
            input_path = os.path.join(raw_dir, f"{split_name}_data.pickle")
            cache_suffix = "_no_topo" if self.skip_topology_validation else ""
            cache_suffix += "_canon" if self.canonicalize_atom_order else ""
            output_path = os.path.join(processed_dir, f"{split_name}{cache_suffix}.npz")
            
            print(f"\nProcessing {split_name} split...")
            
            with open(input_path, "rb") as f:
                data = pickle.load(f)
            
            initial_size = len(data)
            initial_conformer_count = sum(len(mols) for _, mols in data)
            
            # Prepare arguments for parallel processing
            args_list = [
                (smiles, mols, self.skip_topology_validation, self.topology_tolerance, self.ATOMIC_NUMBERS, self.canonicalize_atom_order)
                for smiles, mols in data
            ]
            
            # Process in parallel
            n_workers = min(multiprocessing.cpu_count(), 16)
            print(f"  Using {n_workers} workers...")
            
            all_positions = []
            all_species = []
            all_atom_types = []
            all_n_atoms = []
            all_smiles = []
            all_mol_indices = []
            
            mol_idx = 0
            conformer_counts = []
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(tqdm.tqdm(
                    executor.map(_process_single_molecule_entry, args_list, chunksize=100),
                    total=len(args_list),
                    desc=f"Processing {split_name}"
                ))
            
            # Collect results
            for result in results:
                if result is None:
                    continue
                
                for conf_data in result:
                    all_positions.append(conf_data['positions'])
                    all_species.append(conf_data['species'])
                    all_atom_types.append(utils.atomic_numbers_to_symbols(conf_data['atomic_numbers']))
                    all_n_atoms.append(len(conf_data['positions']))
                    all_smiles.append(conf_data['smiles'])
                    all_mol_indices.append(mol_idx)
                
                conformer_counts.append(len(result))
                mol_idx += 1
            
            # Report statistics
            final_size = mol_idx
            removed_molecules = initial_size - final_size
            kept_conformers = len(all_positions)
            dropped_conformers = initial_conformer_count - kept_conformers
            
            print(f"  Molecules: {final_size} kept, {removed_molecules} removed ({removed_molecules/initial_size*100:.1f}%)")
            print(f"  Conformers: {kept_conformers} kept, {dropped_conformers} removed ({dropped_conformers/initial_conformer_count*100:.1f}%)")
            
            if conformer_counts:
                print(f"  Conformers per molecule: min={min(conformer_counts)}, max={max(conformer_counts)}, mean={np.mean(conformer_counts):.1f}")
            
            # Save to numpy cache
            np.savez(
                output_path,
                positions=np.array(all_positions, dtype=object),
                species=np.array(all_species, dtype=object),
                atom_types=np.array(all_atom_types, dtype=object),
                n_atoms=np.array(all_n_atoms, dtype=np.int32),
                smiles=np.array(all_smiles, dtype=object),
                mol_indices=np.array(all_mol_indices, dtype=np.int32),
            )
            print(f"  Saved to {output_path}")
        
        print("Preprocessing complete!")

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
        
        return datatypes.Graph(
            nodes=dict(
                positions=self._positions[real_idx],
                species=self._species[real_idx],
                atom_types=self._atom_types[real_idx],
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
