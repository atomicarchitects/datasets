"""Utilities for downloading and extracting datasets."""

import os
from typing import Optional

import tqdm
import sh
import git
import zipfile
import tarfile
import urllib
import urllib.request


def clone_url(url: str, root: str) -> str:
    """Clone git repo if it does not exist in root already. Returns path to repo."""
    repo_path = os.path.join(root, url.rpartition("/")[-1].rpartition(".")[0])

    if os.path.exists(repo_path):
        print(f"Using cloned repo: {repo_path}")
        return repo_path

    print(f"Cloning {url} to {repo_path}")
    git.Repo.clone_from(url, repo_path)
    print(f"Cloned {url} to {repo_path}")

    return repo_path


def download_url(url: str, root: str, filename: Optional[str] = None) -> str:
    """Download if file does not exist in root already. Returns path to file."""
    if not filename:
        filename = url.rpartition("/")[2]
    file_path = os.path.join(root, filename)

    if os.path.exists(file_path):
        print(f"Using downloaded file: {file_path}")
        return file_path

    try:
        data = urllib.request.urlopen(url)
    except urllib.error.URLError:
        # No internet connection
        if os.path.exists(file_path):
            print(f"No internet connection! Using downloaded file: {file_path}")
            return file_path

        raise ValueError(f"Could not download {url}")

    chunk_size = 1024
    total_size = int(data.info()["Content-Length"].strip())

    if os.path.exists(file_path):
        if os.path.getsize(file_path) == total_size:
            print(f"Using downloaded and verified file: {file_path}")
            return file_path

    print(f"Downloading {url} to {file_path}")
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
    print(f"Extracting {path} to {root}")
    with zipfile.ZipFile(path, "r") as f:
        for name in f.namelist():
            if name.endswith("/"):
                print(f"Skip directory {name}")
                continue
            out_path = os.path.join(root, name)
            file_size = f.getinfo(name).file_size
            if os.path.exists(out_path) and os.path.getsize(out_path) == file_size:
                print(f"Skip existing file {name}")
                continue
            print(f"Extracting {name} to {root}")
            f.extract(name, root)


def extract_tar(path: str, root: str):
    """Extract a .tar file to the root."""
    print(f"Extracting {path} to {root}")
    with tarfile.open(path, "r") as f:
        f.extractall(path=root)


def extract_gz(path: str) -> str:
    """Extract a .gz file, and return the path to the extracted file."""
    print(f"Unzipping {path}")
    sh.gunzip(path)
    # Remove the .gz extension
    return path[:-3]
