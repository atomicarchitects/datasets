import os
from typing import Iterable, Dict, Optional

import pickle
import time
import hashlib
import glob
import inspect
from functools import wraps


def get_file_hash(filepath: str) -> str:
    """Get SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:8]  # Use first 8 characters
    except FileNotFoundError:
        # If file not found, return empty string (might be running in REPL)
        return ""


def cache_to_file(dataset: str, cache_dir: Optional[str] = None):
    """Decorator to cache the results of load_dataset to a file."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract arguments
            root_dir = args[0] if args else kwargs.get("root_dir")
            check_with_rdkit = (
                args[1] if len(args) > 1 else kwargs.get("check_with_rdkit", True)
            )
            check_validity = (
                args[2] if len(args) > 2 else kwargs.get("check_validity", True)
            )
            start_index = args[3] if len(args) > 3 else kwargs.get("start_index", None)
            end_index = args[4] if len(args) > 4 else kwargs.get("end_index", None)

            # Create cache directory
            cache_path = cache_dir or os.path.join(root_dir, "cache")
            os.makedirs(cache_path, exist_ok=True)

            # Get hash of current file for cache invalidation
            current_file = inspect.getfile(inspect.currentframe())
            file_hash = get_file_hash(current_file)

            # Create cache filename based on parameters and file hash
            cache_name = f"{dataset}_cache_{check_with_rdkit}_{check_validity}_{start_index}_{end_index}_{file_hash}.pkl"
            cache_file = os.path.join(cache_path, cache_name)

            # Check if cache exists and load if available
            if os.path.exists(cache_file):
                print(f"Loading cached {dataset} data from {os.path.abspath(cache_file)}")
                with open(cache_file, "rb") as f:
                    cached_graphs = pickle.load(f)
                for graph in cached_graphs:
                    yield graph
                return

            # Remove any old cache files with different hashes
            old_cache_pattern = (
                f"{dataset}_cache_{check_with_rdkit}_{check_validity}_{start_index}_{end_index}_*.pkl"
            )
            for old_cache in glob.glob(os.path.join(cache_path, old_cache_pattern)):
                if os.path.basename(old_cache) != os.path.basename(cache_file):
                    print(f"Removing outdated cache file: {old_cache}")
                    try:
                        os.remove(old_cache)
                    except OSError:
                        pass

            # If cache doesn't exist, run the original function and cache results
            print(f"Caching {dataset} data to {os.path.abspath(cache_file)}")
            start_time = time.time()
            graphs = list(func(*args, **kwargs))
            end_time = time.time()

            # Save to cache
            with open(cache_file, "wb") as f:
                pickle.dump(graphs, f)

            print(
                f"Cached {len(graphs)} molecules in {end_time - start_time:.2f} seconds"
            )

            # Yield the results
            for graph in graphs:
                yield graph

        return wrapper

    return decorator