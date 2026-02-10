#!/usr/bin/env python
"""
Preprocessing script for GEOM-Drugs dataset.

This script processes raw pickle files and outputs memory-mapped numpy arrays
ready for fast loading. The outputs can be uploaded to Zenodo for distribution.

Usage:
    python preprocess.py --raw-dir ./raw/geom --output-dir ./processed/geom

Output structure (ready for Zenodo upload):
    processed/
    └── geom-drugs/
        ├── train_positions.npy
        ├── train_species.npy
        ├── train_atom_types.npy
        ├── train_offsets.npy
        ├── train_n_atoms.npy
        ├── train_mol_indices.npy
        ├── train_atom_type_lookup.npy
        ├── train_smiles.json
        ├── val_*.npy / val_*.json
        ├── test_*.npy / test_*.json
        ├── manifest.json
        └── README.md
"""

import argparse
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tqdm

from rdkit import Chem
from rdkit.Chem import SanitizeFlags


# =============================================================================
# Utilities
# =============================================================================

ATOMIC_SYMBOLS = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
    33: 'As', 35: 'Br', 53: 'I', 80: 'Hg', 83: 'Bi'
}

GEOM_ATOMIC_NUMBERS = np.array([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83])

COVALENT_RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
}


def atomic_numbers_to_symbols(atomic_numbers: np.ndarray) -> np.ndarray:
    """Convert atomic numbers to element symbols."""
    return np.array([ATOMIC_SYMBOLS.get(n, f'X{n}') for n in atomic_numbers])


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


# =============================================================================
# GEOM-Drugs preprocessing
# =============================================================================


def _process_molecule(mol):
    """Sanitize and kekulize a molecule."""
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
    """Check if bond lengths are within tolerance."""
    adjacency_mask = (adjacency_matrix > 0).astype(int)
    radii = np.array([COVALENT_RADII.get(n, 1.5) for n in numbers])
    ref_dist = (radii[:, np.newaxis] + radii[np.newaxis, :]) * adjacency_mask
    
    diff = coordinates[:, :, np.newaxis, :] - coordinates[:, np.newaxis, :, :]
    data_dist = np.linalg.norm(diff, axis=-1) * adjacency_mask
    
    diffs = np.abs(data_dist - ref_dist[np.newaxis, :, :]) <= (ref_dist[np.newaxis, :, :] * tolerance)
    return diffs.all(axis=(1, 2))


def _validate_topology(mol, tolerance=0.4):
    """Validate topology."""
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    
    conformers = mol.GetConformers()
    if not conformers:
        return False
    
    coordinates = np.array([conf.GetPositions() for conf in conformers])
    return _check_topology(adjacency_matrix, numbers, coordinates, tolerance).all()


def _process_geom_entry(args):
    """Process a single GEOM molecule entry."""
    smiles, mols, skip_topology, topology_tolerance = args
    
    reference_mol = Chem.MolFromSmiles(smiles)
    if reference_mol is None:
        return None
    
    valid_conformers = []
    for mol in mols:
        sanitized = _process_molecule(mol)
        if sanitized is None:
            continue
        if skip_topology or _validate_topology(sanitized, topology_tolerance):
            valid_conformers.append(sanitized)
    
    if not valid_conformers:
        return None
    
    conformer_data = []
    for mol in valid_conformers:
        conformer = mol.GetConformer(0)
        positions = np.array(conformer.GetPositions(), dtype=np.float32)
        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        species = np.searchsorted(GEOM_ATOMIC_NUMBERS, atomic_numbers).astype(np.int32)
        smiles_out = Chem.MolToSmiles(mol, allHsExplicit=True)
        
        conformer_data.append({
            'positions': positions,
            'atomic_numbers': atomic_numbers,
            'species': species,
            'smiles': smiles_out,
        })
    
    return conformer_data


def save_split(
    output_dir: str,
    prefix: str,
    all_positions: List[np.ndarray],
    all_species: List[np.ndarray],
    all_atom_types: List[np.ndarray],
    all_n_atoms: List[int],
    all_smiles: List[str],
    all_mol_indices: List[int],
) -> Dict[str, str]:
    """Save a split in memory-mapped format. Returns file hashes."""
    base = os.path.join(output_dir, prefix)
    file_hashes = {}
    
    # Flatten variable-length arrays
    flat_positions = np.concatenate(all_positions, axis=0).astype(np.float32)
    flat_species = np.concatenate(all_species, axis=0).astype(np.int32)
    
    # Encode atom types as integers
    unique_atom_types = sorted(set(at for types in all_atom_types for at in types))
    atom_type_to_idx = {at: i for i, at in enumerate(unique_atom_types)}
    flat_atom_types = np.concatenate([
        np.array([atom_type_to_idx[at] for at in types], dtype=np.int32)
        for types in all_atom_types
    ])
    
    # Compute offsets
    lengths = np.array(all_n_atoms, dtype=np.int32)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    
    # Save arrays
    files_to_save = {
        f"{prefix}_positions.npy": flat_positions,
        f"{prefix}_species.npy": flat_species,
        f"{prefix}_atom_types.npy": flat_atom_types,
        f"{prefix}_offsets.npy": offsets,
        f"{prefix}_n_atoms.npy": lengths,
        f"{prefix}_mol_indices.npy": np.array(all_mol_indices, dtype=np.int32),
        f"{prefix}_atom_type_lookup.npy": np.array(unique_atom_types, dtype=object),
    }
    
    for filename, arr in files_to_save.items():
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, arr)
        file_hashes[filename] = compute_file_hash(filepath)
    
    # Save SMILES as JSON
    smiles_file = f"{prefix}_smiles.json"
    smiles_path = os.path.join(output_dir, smiles_file)
    with open(smiles_path, 'w') as f:
        json.dump(all_smiles, f)
    file_hashes[smiles_file] = compute_file_hash(smiles_path)
    
    return file_hashes


def preprocess_geom_drugs(
    raw_dir: str,
    output_dir: str,
    skip_topology: bool = False,
    topology_tolerance: float = 0.4,
    n_workers: int = 8,
) -> Dict:
    """
    Preprocess GEOM-Drugs dataset.
    
    Expects raw_dir to contain:
        - train_data.pickle
        - val_data.pickle
        - test_data.pickle
    """
    print("=" * 60)
    print("Preprocessing GEOM-Drugs")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_file_hashes = {}
    stats = {}
    
    for split_name in ["train", "val", "test"]:
        pickle_path = os.path.join(raw_dir, f"{split_name}_data.pickle")
        
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(
                f"Missing {pickle_path}. Download from: "
                "https://bits.csb.pitt.edu/files/geom_raw/"
            )
        
        print(f"\nProcessing {split_name} split...")
        print(f"  Loading {pickle_path}...")
        
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        
        initial_molecules = len(data)
        initial_conformers = sum(len(mols) for _, mols in data)
        print(f"  Loaded {initial_molecules:,} molecules, {initial_conformers:,} conformers")
        
        # Prepare args for parallel processing
        args_list = [
            (smiles, mols, skip_topology, topology_tolerance)
            for smiles, mols in data
        ]
        
        # Free memory
        del data
        
        # Process in parallel
        print(f"  Processing with {n_workers} workers...")
        
        all_positions = []
        all_species = []
        all_atom_types = []
        all_n_atoms = []
        all_smiles = []
        all_mol_indices = []
        
        mol_idx = 0
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm.tqdm(
                executor.map(_process_geom_entry, args_list, chunksize=100),
                total=len(args_list),
                desc=f"  {split_name}"
            ))
        
        for result in results:
            if result is None:
                continue
            
            for conf_data in result:
                all_positions.append(conf_data['positions'])
                all_species.append(conf_data['species'])
                all_atom_types.append(atomic_numbers_to_symbols(conf_data['atomic_numbers']))
                all_n_atoms.append(len(conf_data['positions']))
                all_smiles.append(conf_data['smiles'])
                all_mol_indices.append(mol_idx)
            
            mol_idx += 1
        
        final_molecules = mol_idx
        final_conformers = len(all_positions)
        
        print(f"  Kept {final_molecules:,} molecules ({final_molecules/initial_molecules*100:.1f}%)")
        print(f"  Kept {final_conformers:,} conformers ({final_conformers/initial_conformers*100:.1f}%)")
        
        # Save
        file_hashes = save_split(
            output_dir=output_dir,
            prefix=split_name,
            all_positions=all_positions,
            all_species=all_species,
            all_atom_types=all_atom_types,
            all_n_atoms=all_n_atoms,
            all_smiles=all_smiles,
            all_mol_indices=all_mol_indices,
        )
        
        all_file_hashes.update(file_hashes)
        stats[split_name] = {
            "molecules": final_molecules,
            "conformers": final_conformers,
        }
    
    # Generate README
    readme = generate_geom_readme(stats, all_file_hashes)
    readme_path = os.path.join(os.path.dirname(os.path.normpath(output_dir)), "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    # Save manifest
    manifest = {
        "dataset": "GEOM-Drugs",
        "generated": datetime.now().isoformat(),
        "stats": stats,
        "files": all_file_hashes,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nSaved to {output_dir}")
    return manifest


def generate_geom_readme(stats: Dict, file_hashes: Dict[str, str]) -> str:
    """Generate README for GEOM-Drugs processed data."""
    total_mols = sum(s["molecules"] for s in stats.values())
    total_confs = sum(s["conformers"] for s in stats.values())
    
    return f"""# GEOM-Drugs Preprocessed Dataset

Preprocessed version of the GEOM-Drugs dataset for fast loading.

## Dataset Info
- **Source**: https://www.nature.com/articles/s41597-022-01288-4
- **Total molecules**: {total_mols:,}
- **Total conformers**: {total_confs:,}
- **Splits**: train ({stats['train']['molecules']:,} mol / {stats['train']['conformers']:,} conf), val ({stats['val']['molecules']:,} mol / {stats['val']['conformers']:,} conf), test ({stats['test']['molecules']:,} mol / {stats['test']['conformers']:,} conf)
- **Generated**: {datetime.now().isoformat()}

## Files

Each split (train/val/test) has the following files:

| File | Description |
|------|-------------|
| {{split}}_positions.npy | Atomic positions (N_total, 3) float32 |
| {{split}}_species.npy | Species indices (N_total,) int32 |
| {{split}}_atom_types.npy | Atom type indices (N_total,) int32 |
| {{split}}_offsets.npy | Conformer start indices (n_conf+1,) int64 |
| {{split}}_n_atoms.npy | Atoms per conformer (n_conf,) int32 |
| {{split}}_mol_indices.npy | Molecule index for each conformer (n_conf,) int32 |
| {{split}}_atom_type_lookup.npy | Index to symbol mapping |
| {{split}}_smiles.json | SMILES strings for each conformer |

## License

Please cite the original GEOM paper when using this data.
"""


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GEOM-Drugs dataset for Zenodo distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Directory containing raw pickle files (train_data.pickle, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for processed output"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--skip-topology",
        action="store_true",
        help="Skip topology validation (faster but less strict)"
    )
    
    args = parser.parse_args()
    
    preprocess_geom_drugs(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        skip_topology=args.skip_topology,
        n_workers=args.n_workers,
    )
    
    print("\n" + "=" * 60)
    print("Done! Files are ready for Zenodo upload.")
    print("=" * 60)


if __name__ == "__main__":
    main()