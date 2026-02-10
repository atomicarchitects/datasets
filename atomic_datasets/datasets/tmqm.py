from typing import Iterable, Dict, Optional, Sequence
import os
import logging

import tqdm
import numpy as np
import pandas as pd
import ase.io

from atomic_datasets import utils
from atomic_datasets import datatypes


TMQM_URL = r"https://github.com/bbskjelstad/tmqm.git"


class tmQM(datatypes.MolecularDataset):
    """
    The tmQM dataset (2024 release) from https://pubs.acs.org/doi/10.1021/acs.jcim.0c01041.
    
    Contains ~108k transition metal complexes with elements from H (1) to Hg (80).
    
    Loaded data includes:
    - Geometries from tmQM_X{1,2,3}.xyz.gz (GFN2-xTB optimized)
    - Per-molecule properties from tmQM_y.csv (DFT level) and XYZ comment lines
    - Per-atom natural charges from tmQM_X.q (DFT NBO analysis)
    
    Args:
        root_dir: Directory to store/load data
        split: Which split to use ('train', 'val', 'test')
        use_default_splits: Use default splits
        splits: Custom split definitions (ignored if use_default_splits=True)
        start_index: Start index for slicing
        end_index: End index for slicing
        rng_seed: Random seed for shuffling
        train_on_single_molecule: If True, use single molecule for all splits
        train_on_single_molecule_index: Index of molecule to use if train_on_single_molecule=True
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
        mmap_mode: Optional[str] = 'r',
    ):
        super().__init__(atomic_numbers=self.ATOMIC_NUMBERS)

        self.root_dir = os.path.join(root_dir, "tmqm")
        self.split = split
        self.start_index = start_index
        self.end_index = end_index
        self.rng_seed = rng_seed
        self.train_on_single_molecule = train_on_single_molecule
        self.train_on_single_molecule_index = train_on_single_molecule_index
        self.mmap_mode = mmap_mode

        if use_default_splits and (splits is not None):
            raise ValueError("If use_default_splits is True, splits must be None.")
        if (not use_default_splits) and (splits is None):
            raise ValueError("If use_default_splits is False, splits must be provided.")

        if use_default_splits:
            self.splits = {
                "train": np.arange(86000),
                "val": np.arange(86000, 97000),
                "test": np.arange(97000, 108735),
            }
        else:
            self.splits = splits

        self._positions = None       # (N_total_atoms, 3) float32
        self._atomic_numbers = None  # (N_total_atoms,) int32
        self._charges = None         # (N_total_atoms,) float32 — NBO natural charges
        self._offsets = None         # (N_mols + 1,) int32
        self._n_atoms = None         # (N_mols,) int32
        self._indices = None         # active indices after split/slicing
        self._properties_df = None   # per-molecule properties (CSV + comment line)

        self.preprocess()

    def preprocess(self):
        """Load and preprocess tmQM data."""
        cache_file = os.path.join(self.root_dir, "tmqm_preprocessed.npz")
        prop_cache = cache_file.replace(".npz", ".pkl")
        
        if os.path.exists(cache_file):
            self._load_from_cache(cache_file, prop_cache)
        else:
            self._load_from_raw()
            self._save_to_cache(cache_file, prop_cache)
        
        # Apply splits
        n_molecules = len(self._offsets) - 1
        split_indices = self._get_split_indices(n_molecules)
        
        if self.split is not None:
            self._indices = split_indices[self.split].copy()
        else:
            self._indices = np.arange(n_molecules)
        
        # Apply start/end slicing
        self._indices = self._indices[slice(self.start_index, self.end_index)]

    def _download_if_needed(self) -> str:
        """Download and extract raw XYZ files if not already present, then return the path to the XYZ directory."""
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        xyzs_path = os.path.join(self.root_dir, "xyz")
        if os.path.exists(xyzs_path) and len(os.listdir(xyzs_path)) > 0:
            print(f"Using downloaded data: {os.path.abspath(xyzs_path)}")
            return xyzs_path

        os.makedirs(xyzs_path, exist_ok=True)
        cloned_path = utils.clone_url(TMQM_URL, self.root_dir)

        # 2024 release: 3 XYZ files (tmQM_X1.xyz.gz, tmQM_X2.xyz.gz, tmQM_X3.xyz.gz)
        for i in range(1, 4):
            gz_path = os.path.join(cloned_path, "tmQM", f"tmQM_X{i}.xyz.gz")
            assert os.path.exists(gz_path), f"Expected file not found: {gz_path}"
            mol_file = utils.extract_gz(gz_path)
            with open(mol_file, "r") as f:
                all_xyzs = f.read().split("\n\n")
                for xyz_n, xyz in enumerate(all_xyzs):
                    if xyz.strip() == "":
                        continue
                    with open(os.path.join(xyzs_path, f"X{i}_{xyz_n}.xyz"), "w") as out:
                        out.write(xyz)
        return xyzs_path

    def _load_from_cache(self, cache_file: str, prop_cache: str):
        """Load preprocessed data from cache using memory-mapping."""
        print(f"Loading tmQM from cache: {os.path.abspath(cache_file)}")
        data = np.load(cache_file, mmap_mode=self.mmap_mode)
        
        self._positions = data['positions']
        self._atomic_numbers = data['atomic_numbers']
        self._offsets = data['offsets']
        self._n_atoms = data['n_atoms']
        
        if 'charges' in data:
            self._charges = data['charges']

        if os.path.exists(prop_cache):
            self._properties_df = pd.read_pickle(prop_cache)
            print(f"Loaded properties for {len(self._properties_df)} molecules")
        else:
            logging.warning(f"Properties cache not found: {prop_cache}")

    def _save_to_cache(self, cache_file: str, prop_cache: str):
        """Save preprocessed data to cache."""
        print(f"Saving tmQM cache to: {cache_file}")
        save_dict = dict(
            positions=self._positions,
            atomic_numbers=self._atomic_numbers,
            offsets=self._offsets,
            n_atoms=self._n_atoms,
        )
        if self._charges is not None:
            save_dict['charges'] = self._charges
        np.savez(cache_file, **save_dict)
        
        if self._properties_df is not None:
            self._properties_df.to_pickle(prop_cache)
            print(f"Saved properties cache to: {prop_cache}")

    def _load_properties_csv(self) -> Optional[pd.DataFrame]:
        """Load tmQM_y.csv properties and return as a DataFrame indexed by CSD_code."""
        candidate_paths = [
            os.path.join(self.root_dir, "tmQM_y.csv"),
            os.path.join(self.root_dir, "tmqm", "tmQM_y.csv"),
        ]
        if os.path.exists(self.root_dir):
            for subdir in os.listdir(self.root_dir):
                candidate_paths.append(
                    os.path.join(self.root_dir, subdir, "tmQM", "tmQM_y.csv")
                )
        
        props_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                props_path = p
                break
        
        if props_path is None:
            logging.warning(
                "tmQM_y.csv not found. CSV properties will not be available. "
                f"Searched: {candidate_paths}"
            )
            return None
        
        print(f"Loading properties from: {props_path}")
        df = pd.read_csv(props_path, sep=";")
        df.columns = df.columns.str.strip()
        df["CSD_code"] = df["CSD_code"].str.strip()
        df = df.set_index("CSD_code")
        if "SMILES" in df.columns:
            df = df.rename(columns={"SMILES": "smiles"})
        return df

    def _load_charges_file(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Load tmQM_X.q — per-atom NBO natural charges.
        
        File format (blocks separated by blank lines):
            CSD_code = WELROW | 2020-2024 CSD
            La   0.43487
            Se  -0.38460
            ...
        
        Returns a dict mapping CSD_code -> np.ndarray of charges (float32).
        """
        candidate_paths = [
            os.path.join(self.root_dir, "tmQM_X.q"),
            os.path.join(self.root_dir, "tmqm", "tmQM_X.q"),
        ]
        if os.path.exists(self.root_dir):
            for subdir in os.listdir(self.root_dir):
                candidate_paths.append(
                    os.path.join(self.root_dir, subdir, "tmQM", "tmQM_X.q")
                )
        
        q_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                q_path = p
                break
        
        if q_path is None:
            logging.warning(
                "tmQM_X.q not found. Per-atom charges will not be available. "
                f"Searched: {candidate_paths}"
            )
            return None
        
        print(f"Loading per-atom charges from: {q_path}")
        charges_dict = {}
        
        with open(q_path, "r") as f:
            content = f.read()
        
        blocks = content.split("\n\n")
        for block in tqdm.tqdm(blocks, desc="Parsing tmQM_X.q"):
            block = block.strip()
            if not block:
                continue
            
            lines = block.split("\n")
            # First line is the header: "CSD_code = WELROW | 2020-2024 CSD"
            header = lines[0]
            csd_code = self._parse_csd_code_from_header(header)
            
            # Remaining lines: "Element  charge"
            mol_charges = []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mol_charges.append(float(parts[1]))
                    except ValueError:
                        continue
            
            if csd_code and mol_charges:
                charges_dict[csd_code] = np.array(mol_charges, dtype=np.float32)
        
        print(f"Loaded charges for {len(charges_dict)} molecules from tmQM_X.q")
        return charges_dict

    @staticmethod
    def _parse_comment_line(comment_line: str) -> Dict[str, str]:
        """
        Parse the XYZ comment line into a dict of metadata.
        
        Format:
            CSD_code = WELROW | q = 0 | S = 0 | Stoichiometry = C40H36LaN2P3Se6 | MND = 8 | 2020-2024 CSD
        
        Returns dict with keys: CSD_code, q, S, Stoichiometry, MND, CSD_years
        """
        result = {}
        fields = comment_line.split("|")
        for field in fields:
            field = field.strip()
            if "=" in field:
                key, val = field.split("=", 1)
                result[key.strip()] = val.strip()
            elif field:
                # Last field without '=' is typically "2020-2024 CSD"
                result["CSD_years"] = field.strip()
        return result

    @staticmethod
    def _parse_csd_code_from_header(header: str) -> Optional[str]:
        """Extract CSD_code from a header line like 'CSD_code = WELROW | ...'."""
        fields = header.split("|")
        for field in fields:
            field = field.strip()
            if field.startswith("CSD_code") and "=" in field:
                return field.split("=", 1)[1].strip()
        return None

    @staticmethod
    def _extract_comment_line(mol_path: str) -> Optional[str]:
        """Read the comment line (line 2) from an XYZ file."""
        try:
            with open(mol_path, "r") as f:
                f.readline()  # line 1: atom count
                return f.readline().strip()
        except Exception:
            return None

    def _load_from_raw(self):
        """Load from raw XYZ files and convert to contiguous arrays."""
        xyzs_path = self._download_if_needed()
        
        # Load auxiliary data sources
        props_df = self._load_properties_csv()
        charges_dict = self._load_charges_file()
        
        all_positions, all_z, all_charges, n_atoms_list = [], [], [], []
        comment_metadata = []
        csd_codes = []
        xyz_files = sorted(os.listdir(xyzs_path))
        
        for mol_file in tqdm.tqdm(xyz_files, desc="Preprocessing tmQM"):
            mol_path = os.path.join(xyzs_path, mol_file)
            try:
                mol = ase.io.read(mol_path, format="xyz")
            except Exception as e:
                logging.warning(f"Failed to load {mol_file}: {e}")
                continue

            n_at = len(mol.positions)
            all_positions.append(np.asarray(mol.positions, dtype=np.float32))
            all_z.append(np.asarray(mol.numbers, dtype=np.int32))
            n_atoms_list.append(n_at)
            
            # Parse comment line for CSD code + metadata
            comment = self._extract_comment_line(mol_path)
            if comment:
                meta = self._parse_comment_line(comment)
            else:
                meta = {}
            
            csd_code = meta.get("CSD_code")
            csd_codes.append(csd_code)
            comment_metadata.append(meta)
            
            # Per-atom charges from tmQM_X.q
            if charges_dict is not None and csd_code in charges_dict:
                mol_charges = charges_dict[csd_code]
                if len(mol_charges) == n_at:
                    all_charges.append(mol_charges)
                else:
                    logging.warning(
                        f"Charge count mismatch for {csd_code}: "
                        f"expected {n_at}, got {len(mol_charges)}. Filling with NaN."
                    )
                    all_charges.append(np.full(n_at, np.nan, dtype=np.float32))
            else:
                all_charges.append(np.full(n_at, np.nan, dtype=np.float32))
        
        # Build contiguous arrays
        self._positions = np.concatenate(all_positions, axis=0)
        self._atomic_numbers = np.concatenate(all_z, axis=0)
        self._n_atoms = np.array(n_atoms_list, dtype=np.int32)
        
        self._offsets = np.zeros(len(n_atoms_list) + 1, dtype=np.int32)
        np.cumsum(n_atoms_list, out=self._offsets[1:])
        
        # Per-atom charges
        if charges_dict is not None:
            self._charges = np.concatenate(all_charges, axis=0)
        else:
            self._charges = None

        # Build properties DataFrame: comment-line metadata + CSV properties
        self._properties_df = self._build_properties_df(
            csd_codes, comment_metadata, props_df
        )

    @staticmethod
    def _build_properties_df(
        csd_codes: list,
        comment_metadata: list,
        csv_props_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Merge comment-line metadata and CSV properties into a single DataFrame
        aligned to molecule loading order.
        """
        # Start with comment-line metadata (already contains CSD_code from parsing)
        meta_df = pd.DataFrame(comment_metadata)
        
        # Merge with CSV properties if available
        if csv_props_df is not None:
            # Drop comment-line columns that overlap with CSV columns (keep CSV version)
            overlap = set(meta_df.columns) & set(csv_props_df.columns)
            if overlap:
                meta_df = meta_df.drop(columns=list(overlap))
            
            merged = meta_df.merge(
                csv_props_df, left_on="CSD_code", right_index=True,
                how="left",
            )
            
            n_matched = merged[csv_props_df.columns[0]].notna().sum()
            n_total = len(csd_codes)
            print(f"CSV properties matched: {n_matched}/{n_total} molecules")
            
            if n_matched < n_total:
                logging.warning(
                    f"{n_total - n_matched} molecules have no matching CSV properties. "
                    "Their CSV property values will be NaN."
                )
            return merged
        
        return meta_df


    def _get_split_indices(self, n_molecules: int) -> Dict[str, np.ndarray]:
        if self.train_on_single_molecule:
            idx = np.array([self.train_on_single_molecule_index])
            return {"train": idx, "val": idx, "test": idx}

        rng = np.random.default_rng(self.rng_seed)
        shuffled_indices = np.arange(n_molecules)
        rng.shuffle(shuffled_indices)
        
        return {k: shuffled_indices[v[v < n_molecules]] for k, v in self.splits.items()}

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> datatypes.Graph:
        """Fast slice access via offsets (O(1))."""
        real_idx = self._indices[idx]
        start, end = self._offsets[real_idx], self._offsets[real_idx + 1]
        atomic_numbers = np.array(self._atomic_numbers[start:end])
        
        # Build nodes dict
        nodes = dict(
            positions=np.array(self._positions[start:end]),
            atomic_numbers=atomic_numbers,
            species=self.atomic_numbers_to_species(atomic_numbers),
            atom_types=utils.atomic_numbers_to_symbols(atomic_numbers),
        )
        
        # Per-atom NBO natural charges
        if self._charges is not None:
            nodes["charges"] = np.array(self._charges[start:end])
        
        # Per-molecule properties
        properties = None
        if self._properties_df is not None:
            properties = self._properties_df.iloc[real_idx].to_dict()
        
        return datatypes.Graph(
            nodes=nodes,
            n_node=np.asarray([self._n_atoms[real_idx]]),
            edges=None, receivers=None, senders=None, globals=None, n_edge=None,
            properties=properties,
        )