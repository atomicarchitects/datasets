#!/usr/bin/env python
"""
Preprocessing script for CrossDocked dataset.

This script downloads the raw data, processes it using SBDD splits,
and outputs memory-mapped numpy arrays ready for fast loading.

Usage:
    python preprocess_crossdocked.py --root-dir ./data/crossdocked --output-dir ./processed/crossdocked

Output structure:
    processed/
    └── crossdocked/
        ├── train_positions.npy
        ├── train_atom_types.npy
        ├── train_offsets.npy
        ├── train_n_atoms.npy
        ├── train_atom_type_lookup.npy
        ├── train_properties.json
        ├── val_*.npy / val_*.json
        ├── test_*.npy / test_*.json
        ├── manifest.json
        └── README.md
"""

import argparse
import os
import json
import hashlib
import subprocess
import tarfile
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import tqdm
import rdkit.Chem as Chem

from atomic_datasets import utils

CROSSDOCKED_URL="https://ndownloader.figshare.com/articles/25878871/versions/3"
SPLIT_URL="https://drive.google.com/uc?export=download&id=1mycOKpphVBQjxEbpn1AwdpQs8tNVbxKY"



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


# =============================================================================
# Download and extract
# =============================================================================

def ensure_downloaded(root_dir: str):
    """Download and extract raw CrossDocked data and splits if not present."""
    os.makedirs(root_dir, exist_ok=True)

    raw_mols_path = os.path.join(root_dir, "crossdocked_pocket10_with_protein")
    if not os.path.exists(raw_mols_path):
        print(f"Downloading CrossDocked dataset to {root_dir}")
        
        # Figshare serves a .zip containing crossdocked_pocket10_with_protein.tar.gz
        zip_path = utils.download_url(
            CROSSDOCKED_URL, root_dir, "crossdocked_pocket10_with_protein.zip"
        )
        print(f"Downloaded to {zip_path}")
        
        # Step 1: Extract zip (use system unzip for large files)
        print("Extracting zip...")
        result = subprocess.run(
            ["unzip", "-o", zip_path, "-d", root_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"unzip failed: {result.stderr}")
        
        # Step 2: Extract the inner tar.gz
        tar_gz_path = os.path.join(root_dir, "crossdocked_pocket10_with_protein.tar.gz")
        assert os.path.exists(tar_gz_path), (
            f"Expected {tar_gz_path} inside zip. "
            f"Contents of {root_dir}: {os.listdir(root_dir)}"
        )
        print(f"Extracting {tar_gz_path}...")
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            tar.extractall(path=root_dir)
        
        print(f"Extracted to {raw_mols_path}")
    else:
        print(f"Raw data already exists: {raw_mols_path}")

    split_path = os.path.join(root_dir, "split_by_name.pt")
    if not os.path.exists(split_path):
        print(f"Downloading SBDD splits to {root_dir}")
        utils.download_url(SPLIT_URL, root_dir, "split_by_name.pt")
    else:
        print(f"Splits already exist: {split_path}")


# =============================================================================
# Load raw data
# =============================================================================

def load_raw_crossdocked(root_dir: str):
    """Load raw CrossDocked data from SDF/PDB files."""
    data_dir = os.path.join(root_dir, "crossdocked_pocket10_with_protein")

    for pocket in tqdm.tqdm(os.listdir(data_dir), desc="Loading CrossDocked"):
        pocket_dir = os.path.join(data_dir, pocket)
        if not os.path.isdir(pocket_dir):
            continue

        files = os.listdir(pocket_dir)
        pdb_ids = set(['_'.join(x.split('_')[:2]) for x in files])

        for pdb_id in pdb_ids:
            file_pairs = []
            for f in files:
                if not (len(pdb_id) < len(f) and f[:len(pdb_id)] == pdb_id):
                    continue
                if f.endswith(".sdf"):
                    ligand_file = os.path.join(pocket, f)
                    pocket_file = ligand_file[:-4] + "_pocket10.pdb"
                    file_pairs.append((pocket_file, ligand_file))

            for pocket_file, ligand_file in file_pairs:
                try:
                    protein = Chem.MolFromPDBFile(
                        os.path.join(data_dir, pocket_file),
                        sanitize=False, removeHs=False,
                    )
                    with Chem.SDMolSupplier(
                        os.path.join(data_dir, ligand_file),
                        sanitize=False, removeHs=False,
                    ) as suppl:
                        ligand = next(suppl)
                    target = Chem.CombineMols(protein, ligand)
                except Exception as e:
                    print(f"Skipping {ligand_file}: {e}")
                    continue

                atomic_numbers = np.array([atom.GetAtomicNum() for atom in target.GetAtoms()])
                positions = np.array(target.GetConformer().GetPositions(), dtype=np.float32)
                atom_types = utils.atomic_numbers_to_symbols(atomic_numbers)
                starting_fragment_mask = (
                    [1] * protein.GetNumAtoms() + [0] * ligand.GetNumAtoms()
                )

                yield {
                    "positions": positions,
                    "atom_types": atom_types,
                    "pocket_file": pocket_file,
                    "starting_fragment_mask": starting_fragment_mask,
                }


# =============================================================================
# Save split
# =============================================================================

def save_split(
    output_dir: str,
    prefix: str,
    graphs: list,
    type_to_idx: Dict[str, int],
    atom_type_lookup: np.ndarray,
) -> Dict[str, str]:
    """Save a split in memory-mapped format. Returns file hashes."""
    file_hashes = {}

    positions_list = []
    atom_type_indices_list = []
    n_atoms_list = []
    offsets = [0]
    properties_list = []

    for g in graphs:
        n = len(g["positions"])
        positions_list.append(g["positions"])
        atom_type_indices_list.append(
            np.array([type_to_idx[a] for a in g["atom_types"]], dtype=np.int32)
        )
        n_atoms_list.append(n)
        offsets.append(offsets[-1] + n)
        properties_list.append({
            "pocket_file": g["pocket_file"],
            "starting_fragment_mask": g["starting_fragment_mask"],
        })

    files_to_save = {
        f"{prefix}_positions.npy": np.concatenate(positions_list, axis=0).astype(np.float32),
        f"{prefix}_atom_types.npy": np.concatenate(atom_type_indices_list, axis=0).astype(np.int32),
        f"{prefix}_n_atoms.npy": np.array(n_atoms_list, dtype=np.int32),
        f"{prefix}_offsets.npy": np.array(offsets, dtype=np.int64),
        f"{prefix}_atom_type_lookup.npy": atom_type_lookup,
    }

    for filename, arr in files_to_save.items():
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, arr)
        file_hashes[filename] = compute_file_hash(filepath)

    props_file = f"{prefix}_properties.json"
    props_path = os.path.join(output_dir, props_file)
    with open(props_path, "w") as f:
        json.dump(properties_list, f)
    file_hashes[props_file] = compute_file_hash(props_path)

    return file_hashes


# =============================================================================
# Main preprocessing
# =============================================================================

def generate_readme(stats: Dict, file_hashes: Dict[str, str]) -> str:
    """Generate README for CrossDocked processed data."""
    total = sum(s["complexes"] for s in stats.values())
    return f"""# CrossDocked Preprocessed Dataset

Preprocessed version of the CrossDocked dataset with SBDD splits for fast loading.

## Dataset Info
- **Source**: https://pubs.acs.org/doi/full/10.1021/acs.jcim.0c00411
- **Splits**: Luo et al. (https://proceedings.neurips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html)
- **Total complexes**: {total:,}
- **Splits**: train ({stats['train']['complexes']:,}), val ({stats['val']['complexes']:,}), test ({stats['test']['complexes']:,})
- **Val split**: 5% of training data (random seed 0)
- **Generated**: {datetime.now().isoformat()}

## Files

Each split (train/val/test) has the following files:

| File | Description |
|------|-------------|
| {{split}}_positions.npy | Atomic positions (N_total, 3) float32 |
| {{split}}_atom_types.npy | Atom type indices (N_total,) int32 |
| {{split}}_offsets.npy | Complex start indices (n_complex+1,) int64 |
| {{split}}_n_atoms.npy | Atoms per complex (n_complex,) int32 |
| {{split}}_atom_type_lookup.npy | Index to symbol mapping |
| {{split}}_properties.json | pocket_file and starting_fragment_mask per complex |

## License

Please cite the original CrossDocked and Luo et al. papers when using this data.
"""


def preprocess_crossdocked(root_dir: str, output_dir: str) -> Dict:
    """Preprocess CrossDocked dataset."""
    print("=" * 60)
    print("Preprocessing CrossDocked")
    print("=" * 60)

    # Download if needed
    ensure_downloaded(root_dir)

    # Load all raw data
    print("\nLoading raw data...")
    all_graphs = list(load_raw_crossdocked(root_dir))
    print(f"Loaded {len(all_graphs)} complexes")

    # Load SBDD splits
    split_data = torch.load(os.path.join(root_dir, "split_by_name.pt"))
    splits = {s: set(x[0] for x in split_data[s]) for s in split_data}

    # Assign graphs to splits
    train_all = []
    test_graphs = []
    for g in all_graphs:
        if g["pocket_file"] in splits["train"]:
            train_all.append(g)
        elif g["pocket_file"] in splits["test"]:
            test_graphs.append(g)

    # Split train into train/val (last 5% as val)
    np.random.seed(0)
    indices = np.arange(len(train_all))
    np.random.shuffle(indices)
    val_size = max(1, len(train_all) // 20)

    train_graphs = [train_all[i] for i in indices[:-val_size]]
    val_graphs = [train_all[i] for i in indices[-val_size:]]

    split_graphs = {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
    }

    print(f"Split sizes: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    # Build global atom type lookup
    atom_type_set = set()
    for g in all_graphs:
        atom_type_set.update(g["atom_types"])
    atom_type_lookup = np.array(sorted(atom_type_set))
    type_to_idx = {t: i for i, t in enumerate(atom_type_lookup)}

    # Save each split
    os.makedirs(output_dir, exist_ok=True)
    all_file_hashes = {}
    stats = {}

    for split_name, graphs in split_graphs.items():
        print(f"\nSaving {split_name} split: {len(graphs)} complexes")
        file_hashes = save_split(output_dir, split_name, graphs, type_to_idx, atom_type_lookup)
        all_file_hashes.update(file_hashes)
        stats[split_name] = {"complexes": len(graphs)}

    # Generate README
    readme = generate_readme(stats, all_file_hashes)
    readme_path = os.path.join(os.path.dirname(os.path.normpath(output_dir)), "README.md")
    with open(readme_path, "w") as f:
        f.write(readme)

    # Save manifest
    manifest = {
        "dataset": "CrossDocked",
        "generated": datetime.now().isoformat(),
        "stats": stats,
        "files": all_file_hashes,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved to {output_dir}")
    return manifest


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CrossDocked dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--root-dir", type=str, required=True,
        help="Directory for raw CrossDocked data (will download if missing)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for processed output",
    )
    args = parser.parse_args()

    preprocess_crossdocked(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("Done! Files are ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()