#!/usr/bin/env python
"""
Preprocessing script for ChEMBL3D dataset.

This script processes raw PyTorch files and outputs numpy arrays
ready for fast loading without torch dependencies.

Usage:
    python preprocess_chembl3d.py --raw-dir ./raw/chembl3d --output-dir ./processed/chembl3d

Output structure:
    processed/chembl3d/
        ├── train_positions.npy
        ├── train_species.npy
        ├── train_offsets.npy
        ├── train_charges.npy
        ├── train_is_aromatic.npy
        ├── train_is_in_ring.npy
        ├── train_hybridization.npy
        ├── train_metadata.json
        ├── val_*.npy / val_*.json
        ├── test_*.npy / test_*.json (for each test variant)
        ├── manifest.json
        └── README.md
"""

import argparse
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import tqdm


# =============================================================================
# Constants
# =============================================================================

ATOMIC_NUMBERS = np.array([
    1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83, 34
], dtype=np.int32)

ATOM_SYMBOLS = np.array([
    'H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi', 'Se'
])

# Map from split name to filename
SPLIT_FILES = {
    'train': 'train_h.pt',
    'val': 'val_h.pt',
    'test_small': 'test_small_h.pt',
    'test_rot_bonds': 'test_rot_bonds_h.pt',
    'test_cremp': 'test_cremp_h.pt',
}


# =============================================================================
# Utilities
# =============================================================================

def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_split(
    output_dir: str,
    prefix: str,
    positions: np.ndarray,
    species: np.ndarray,
    offsets: np.ndarray,
    charges: np.ndarray,
    is_aromatic: np.ndarray,
    is_in_ring: np.ndarray,
    hybridization: np.ndarray,
    chemblids: List[str],
    smiles: List[str],
) -> Dict[str, str]:
    """Save a split as numpy files. Returns file hashes."""
    file_hashes = {}
    
    # Save arrays
    files_to_save = {
        f"{prefix}_positions.npy": positions,
        f"{prefix}_species.npy": species,
        f"{prefix}_offsets.npy": offsets,
        f"{prefix}_charges.npy": charges,
        f"{prefix}_is_aromatic.npy": is_aromatic,
        f"{prefix}_is_in_ring.npy": is_in_ring,
        f"{prefix}_hybridization.npy": hybridization,
    }
    
    for filename, arr in files_to_save.items():
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, arr)
        file_hashes[filename] = compute_file_hash(filepath)
        print(f"    Saved {filename}: {arr.shape}, {arr.dtype}")
    
    # Save metadata as JSON
    metadata_file = f"{prefix}_metadata.json"
    metadata_path = os.path.join(output_dir, metadata_file)
    with open(metadata_path, 'w') as f:
        json.dump({
            "chemblids": chemblids,
            "smiles": smiles,
        }, f)
    file_hashes[metadata_file] = compute_file_hash(metadata_path)
    print(f"    Saved {metadata_file}: {len(chemblids)} entries")
    
    return file_hashes


def process_pt_file(pt_path: str) -> tuple:
    """
    Load and process a single .pt file.
    
    Returns:
        positions, species, offsets, charges, is_aromatic, is_in_ring, 
        hybridization, chemblids, smiles, n_molecules, n_atoms
    """
    import torch
    from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    from rdkit.Chem.rdchem import Mol
    
    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol])
    data, batch = torch.load(pt_path, weights_only=False)
    
    n_molecules = len(batch["pos"]) - 1
    
    # Convert concatenated tensors directly to numpy
    positions = data.pos.numpy().astype(np.float32)
    species = data.x.numpy().astype(np.uint8)
    charges = data.charges.numpy().astype(np.int8)
    is_aromatic = data.is_aromatic.numpy().astype(np.uint8)
    is_in_ring = data.is_in_ring.numpy().astype(np.uint8)
    hybridization = data.hybridization.numpy().astype(np.uint8)
    offsets = batch['pos'].numpy().astype(np.int64)
    
    # Lists are already Python lists
    chemblids = list(data.chemblid)
    smiles = list(data.smiles)
    
    n_atoms = len(positions)
    
    return (
        positions, species, offsets, charges, is_aromatic, 
        is_in_ring, hybridization, chemblids, smiles, 
        n_molecules, n_atoms
    )


# =============================================================================
# ChEMBL3D preprocessing
# =============================================================================

def preprocess_chembl3d(
    raw_dir: str,
    output_dir: str,
    splits: Optional[List[str]] = None,
) -> Dict:
    """
    Preprocess ChEMBL3D dataset.
    
    Args:
        raw_dir: Directory containing chembl3d/chembl3d/processed/*.pt files
        output_dir: Directory for processed output
        splits: List of splits to process (default: all available)
    """
    print("=" * 60)
    print("Preprocessing ChEMBL3D")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find available splits
    pt_dir = os.path.join(raw_dir, "chembl3d", "processed")
    if not os.path.exists(pt_dir):
        raise FileNotFoundError(
            f"Processed directory not found: {pt_dir}. "
            f"Please ensure the raw data is in the expected structure."
        )
    print(f"Found processed directory: {pt_dir}")
    
    if splits is None:
        splits = [name for name, fname in SPLIT_FILES.items() 
                  if os.path.exists(os.path.join(pt_dir, fname))]
    
    print(f"Processing splits: {splits}")
    
    all_file_hashes = {}
    stats = {}
    
    for split_name in splits:
        pt_filename = SPLIT_FILES.get(split_name)
        if pt_filename is None:
            print(f"  Warning: Unknown split '{split_name}', skipping")
            continue
            
        pt_path = os.path.join(pt_dir, pt_filename)
        
        if not os.path.exists(pt_path):
            print(f"  Warning: {pt_path} not found, skipping")
            continue
        
        print(f"\nProcessing {split_name} split...")
        print(f"  Loading {pt_path}...")
        
        (
            positions, species, offsets, charges, is_aromatic,
            is_in_ring, hybridization, chemblids, smiles,
            n_molecules, n_atoms
        ) = process_pt_file(pt_path)
        
        print(f"  Found {n_molecules:,} molecules, {n_atoms:,} atoms")
        
        # Save
        file_hashes = save_split(
            output_dir=output_dir,
            prefix=split_name,
            positions=positions,
            species=species,
            offsets=offsets,
            charges=charges,
            is_aromatic=is_aromatic,
            is_in_ring=is_in_ring,
            hybridization=hybridization,
            chemblids=chemblids,
            smiles=smiles,
        )
        
        all_file_hashes.update(file_hashes)
        stats[split_name] = {
            "molecules": n_molecules,
            "atoms": n_atoms,
        }
    
    # Save shared files
    lookup_path = os.path.join(output_dir, "atom_type_lookup.npy")
    np.save(lookup_path, ATOM_SYMBOLS)
    all_file_hashes["atom_type_lookup.npy"] = compute_file_hash(lookup_path)
    
    atomic_numbers_path = os.path.join(output_dir, "atomic_numbers.npy")
    np.save(atomic_numbers_path, ATOMIC_NUMBERS)
    all_file_hashes["atomic_numbers.npy"] = compute_file_hash(atomic_numbers_path)
    
    # Generate README
    readme = generate_readme(stats, all_file_hashes)
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    # Save manifest
    manifest = {
        "dataset": "ChEMBL3D",
        "version": "1.0.0",
        "generated": datetime.now().isoformat(),
        "atomic_numbers": ATOMIC_NUMBERS.tolist(),
        "atom_symbols": ATOM_SYMBOLS.tolist(),
        "stats": stats,
        "files": all_file_hashes,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Saved to {output_dir}")
    print(f"{'=' * 60}")
    
    return manifest


def generate_readme(stats: Dict, file_hashes: Dict[str, str]) -> str:
    """Generate README for processed data."""
    total_mols = sum(s["molecules"] for s in stats.values())
    total_atoms = sum(s["atoms"] for s in stats.values())
    
    splits_info = "\n".join(
        f"- **{name}**: {s['molecules']:,} molecules, {s['atoms']:,} atoms"
        for name, s in stats.items()
    )
    
    return f"""# ChEMBL3D Preprocessed Dataset

Preprocessed version of the ChEMBL3D dataset for fast loading without PyTorch dependencies.

## Dataset Info

- **Source**: https://github.com/isayevlab/LoQI
- **Total molecules**: {total_mols:,}
- **Total atoms**: {total_atoms:,}
- **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Splits

{splits_info}

## Atomic Species

| Index | Symbol | Atomic Number |
|-------|--------|---------------|
| 0 | H | 1 |
| 1 | B | 5 |
| 2 | C | 6 |
| 3 | N | 7 |
| 4 | O | 8 |
| 5 | F | 9 |
| 6 | Al | 13 |
| 7 | Si | 14 |
| 8 | P | 15 |
| 9 | S | 16 |
| 10 | Cl | 17 |
| 11 | As | 33 |
| 12 | Br | 35 |
| 13 | I | 53 |
| 14 | Hg | 80 |
| 15 | Bi | 83 |
| 16 | Se | 34 |

## Files

Each split has the following files:

| File | Description |
|------|-------------|
| `{{split}}_positions.npy` | Atomic positions, shape (N_atoms, 3), dtype float32 |
| `{{split}}_species.npy` | Species indices, shape (N_atoms,), dtype uint8 |
| `{{split}}_offsets.npy` | Molecule boundaries, shape (N_mol+1,), dtype int64 |
| `{{split}}_charges.npy` | Formal charges, shape (N_atoms,), dtype int8 |
| `{{split}}_is_aromatic.npy` | Aromaticity flags, shape (N_atoms,), dtype uint8 |
| `{{split}}_is_in_ring.npy` | Ring membership, shape (N_atoms,), dtype uint8 |
| `{{split}}_hybridization.npy` | Hybridization states, shape (N_atoms,), dtype uint8 |
| `{{split}}_metadata.json` | ChEMBL IDs and SMILES strings |

Shared files:
| File | Description |
|------|-------------|
| `atom_type_lookup.npy` | Species index to symbol mapping |
| `atomic_numbers.npy` | Species index to atomic number mapping |
| `manifest.json` | Dataset metadata and file hashes |

## Usage

To get molecule `i`, slice using offsets:
```python
start, end = offsets[i], offsets[i + 1]
mol_positions = positions[start:end]
mol_species = species[start:end]
```

Or use the dataset class:
```python
from atomic_datasets import ChEMBL3D

dataset = ChEMBL3D(root_dir="path/to/processed", split="train")
mol = dataset[0]
```

## File Hashes (SHA256)

```
{chr(10).join(f"{k}: {v}" for k, v in sorted(file_hashes.items()))}
```

## License

Please cite the original LoQI paper when using this data.
"""


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ChEMBL3D dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Directory containing chembl3d/chembl3d/processed/*.pt files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for processed output"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Splits to process (default: all available). Options: train, val, test_small, test_rot_bonds, test_cremp"
    )
    
    args = parser.parse_args()
    
    preprocess_chembl3d(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()