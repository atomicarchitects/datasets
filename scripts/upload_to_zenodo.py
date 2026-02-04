#!/usr/bin/env python
"""
Upload files to Zenodo via REST API.

Usage:
    # First, get your API token from https://zenodo.org/account/settings/applications/
    export ZENODO_TOKEN="your_token_here"
    
    python upload_to_zenodo.py --file ./processed/geom/geom_drugs_processed.zip \
        --title "GEOM-Drugs Preprocessed Dataset" \
        --description "Preprocessed GEOM-Drugs dataset for fast loading."

    # For testing, use sandbox (sandbox.zenodo.org)
    python upload_to_zenodo.py --sandbox --file ./processed/geom/geom_drugs_processed.zip ...
"""

import argparse
import os
import json
import requests
from tqdm import tqdm
from pathlib import Path


ZENODO_API = "https://zenodo.org/api"
ZENODO_SANDBOX_API = "https://sandbox.zenodo.org/api"


def create_deposition(token: str, base_url: str) -> dict:
    """Create a new empty deposition."""
    r = requests.post(
        f"{base_url}/deposit/depositions",
        params={"access_token": token},
        json={},
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.json()


def upload_file(token: str, bucket_url: str, filepath: str) -> dict:
    """Upload a file to a deposition bucket."""
    filename = os.path.basename(filepath)
    filesize = os.path.getsize(filepath)
    
    print(f"Uploading {filename} ({filesize / 1e9:.2f} GB)... This may take a while.")
    
    with open(filepath, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{filename}",
            params={"access_token": token},
            data=f,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(filesize),
            },
        )
    
    r.raise_for_status()
    return r.json()


def update_metadata(token: str, base_url: str, deposition_id: int, metadata: dict) -> dict:
    """Update deposition metadata."""
    r = requests.put(
        f"{base_url}/deposit/depositions/{deposition_id}",
        params={"access_token": token},
        json={"metadata": metadata},
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.json()


def publish_deposition(token: str, base_url: str, deposition_id: int) -> dict:
    """Publish the deposition (makes it permanent!)."""
    r = requests.post(
        f"{base_url}/deposit/depositions/{deposition_id}/actions/publish",
        params={"access_token": token},
    )
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser(
        description="Upload files to Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--file", "-f", required=True, help="File to upload")
    parser.add_argument("--title", "-t", required=True, help="Dataset title")
    parser.add_argument("--description", "-d", required=True, help="Dataset description")
    parser.add_argument("--creators", "-c", nargs="+", default=[], help="Creator names (e.g., 'John Doe')")
    parser.add_argument("--sandbox", action="store_true", help="Use Zenodo sandbox for testing")
    parser.add_argument("--publish", action="store_true", help="Publish immediately (cannot be undone!)")
    parser.add_argument("--token", help="Zenodo API token (or set ZENODO_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.environ.get("ZENODO_TOKEN")
    if not token:
        print("Error: No API token provided.")
        print("Get one from: https://zenodo.org/account/settings/applications/")
        print("Then: export ZENODO_TOKEN='your_token'")
        return 1
    
    # Check file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return 1
    
    base_url = ZENODO_SANDBOX_API if args.sandbox else ZENODO_API
    env_name = "SANDBOX" if args.sandbox else "PRODUCTION"
    
    print(f"Using Zenodo {env_name}: {base_url}")
    print()
    
    # Step 1: Create deposition
    print("Creating deposition...")
    deposition = create_deposition(token, base_url)
    deposition_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]
    print(f"  Deposition ID: {deposition_id}")
    print()
    
    # Step 2: Upload file
    print("Uploading file...")
    upload_file(token, bucket_url, args.file)
    print()
    
    # Step 3: Set metadata
    print("Setting metadata...")
    
    creators = [{"name": name} for name in args.creators] if args.creators else [{"name": "Anonymous"}]
    
    metadata = {
        "title": args.title,
        "upload_type": "dataset",
        "description": args.description,
        "creators": creators,
        "access_right": "open",
        "license": "cc-by-4.0",
    }
    
    update_metadata(token, base_url, deposition_id, metadata)
    print("  Metadata updated")
    print()
    
    # Step 4: Publish (optional)
    if args.publish:
        print("Publishing...")
        result = publish_deposition(token, base_url, deposition_id)
        doi = result.get("doi", "N/A")
        record_url = result["links"]["record_html"]
        file_url = result["files"][0]["links"]["self"]
        
        print()
        print("=" * 60)
        print("PUBLISHED!")
        print("=" * 60)
        print(f"  DOI: {doi}")
        print(f"  Record: {record_url}")
        print(f"  File URL: {file_url}")
        print()
        print("Update your code with:")
        print(f'  GEOM_DRUGS_ZENODO_URL = "{file_url}"')
    else:
        edit_url = f"{'https://sandbox.zenodo.org' if args.sandbox else 'https://zenodo.org'}/deposit/{deposition_id}"
        print("=" * 60)
        print("DRAFT CREATED (not published)")
        print("=" * 60)
        print(f"  Edit/publish at: {edit_url}")
        print()
        print("To publish via CLI, add --publish flag")
        print("WARNING: Publishing is permanent and cannot be undone!")
    
    return 0


if __name__ == "__main__":
    exit(main())