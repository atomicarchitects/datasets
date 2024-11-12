"""Utilities for downloading and extracting datasets."""

from typing import Dict
import os
import logging

import ase
import numpy as np
import jraph
import tqdm
import git
import zipfile
import tarfile
import urllib
from rdkit import Chem


def clone_url(url: str, root: str) -> str:
    """Clone if repo does not exist in root already. Returns path to repo."""
    repo_path = os.path.join(root, url.rpartition("/")[-1].rpartition(".")[0])

    if os.path.exists(repo_path):
        logging.info(f"Using cloned repo: {repo_path}")
        return repo_path

    logging.info(f"Cloning {url} to {repo_path}")
    git.Repo.clone_from(url, repo_path)

    return repo_path


def download_url(url: str, root: str) -> str:
    """Download if file does not exist in root already. Returns path to file."""
    filename = url.rpartition("/")[2]
    file_path = os.path.join(root, filename)

    try:
        if os.path.exists(file_path):
            logging.info(f"Using downloaded file: {file_path}")
            return file_path
        data = urllib.request.urlopen(url)
    except urllib.error.URLError:
        # No internet connection
        if os.path.exists(file_path):
            logging.info(f"No internet connection! Using downloaded file: {file_path}")
            return file_path

        raise ValueError(f"Could not download {url}")

    chunk_size = 1024
    total_size = int(data.info()["Content-Length"].strip())

    if os.path.exists(file_path):
        if os.path.getsize(file_path) == total_size:
            logging.info(f"Using downloaded and verified file: {file_path}")
            return file_path

    logging.info(f"Downloading {url} to {file_path}")

    with open(file_path, "wb") as f:
        with tqdm.tqdm(total=total_size) as pbar:
            while True:
                chunk = data.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(chunk_size)

    return file_path


def extract_zip(path: str, root: str) -> None:
    """Extract zip if content does not exist in root already."""
    logging.info(f"Extracting {path} to {root}")
    with zipfile.ZipFile(path, "r") as f:
        for name in f.namelist():
            if name.endswith("/"):
                logging.info(f"Skip directory {name}")
                continue
            out_path = os.path.join(root, name)
            file_size = f.getinfo(name).file_size
            if os.path.exists(out_path) and os.path.getsize(out_path) == file_size:
                logging.info(f"Skip existing file {name}")
                continue
            logging.info(f"Extracting {name} to {root}")
            f.extract(name, root)


def extract_tar(path: str, root: str):
    """Extract tar."""
    logging.info(f"Extracting {path} to {root}...")
    with tarfile.TarFile(path, "r") as f:
        f.extractall(path=root)


def ase_atoms_to_structure(molecule: ase.Atoms) -> jraph.GraphsTuple:
    """Converts an ase Atoms object to a jraph.GraphsTuple object."""
    return jraph.GraphsTuple(
        nodes=dict(
            positions=np.asarray(molecule.positions),
            species=molecule.numbers - 1
        ),
        edges=None,
        receivers=None,
        senders=None,
        globals=None,
        n_node=np.asarray([len(molecule.numbers)]),
        n_edge=None,
    )


def is_molecule_sane(mol: Chem.Mol) -> bool:
    """Check that the molecule passes some basic sanity checks from Posebusters.
    Source: https://github.com/maabuu/posebusters/blob/main/posebusters/modules/sanity.py
    """

    errors = Chem.rdmolops.DetectChemistryProblems(
        mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_ALL
    )
    types = [error.GetType() for error in errors]
    num_frags = len(Chem.rdmolops.GetMolFrags(mol, asMols=False, sanitizeFrags=False))

    results = {
        "passes_valence_checks": "AtomValenceException" not in types,
        "passes_kekulization": "AtomKekulizeException" not in types,
        "passes_rdkit_sanity_checks": len(errors) == 0,
        "all_atoms_connected": num_frags <= 1,
    }
    return all(results.values())

