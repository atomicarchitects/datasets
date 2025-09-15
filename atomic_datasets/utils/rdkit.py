import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def atomic_number_to_symbol(atomic_number: int) -> str:
    """Convert atomic number to atomic symbol."""
    ptable = Chem.GetPeriodicTable()
    return ptable.GetElementSymbol(atomic_number)


def atomic_numbers_to_symbols(atomic_numbers: np.ndarray) -> np.ndarray:
    """Convert atomic numbers to atomic symbols."""
    return np.array([atomic_number_to_symbol(int(num)) for num in atomic_numbers])


def atomic_symbol_to_number(atomic_symbol: str) -> int:
    """Convert atomic symbol to atomic number."""
    ptable = Chem.GetPeriodicTable()
    return ptable.GetAtomicNumber(atomic_symbol)


def atomic_symbols_to_numbers(atomic_symbols: np.ndarray) -> np.ndarray:
    """Convert atomic symbols to atomic numbers."""
    return np.array([atomic_symbol_to_number(sym) for sym in atomic_symbols])


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

def add_bonds(mol: Chem.Mol) -> Chem.Mol:
    """Adds bonds to a molecule."""
    mol = Chem.RWMol(mol)
    rdDetermineBonds.DetermineBonds(
        mol, charge=0, useHueckel=False, allowChargedFragments=True
    )
    return mol

def check_molecule_validity(mol: Chem.Mol) -> bool:
    """Checks whether a molecule is valid using xyz2mol.

    This function checks whether xyz2mol can determine all bonds in a molecule, with a net charge of 0.
    """
    # Make a copy of the molecule.
    mol = Chem.Mol(mol)

    # We should only have one conformer.
    assert mol.GetNumConformers() == 1

    try:
        Chem.SanitizeMol(mol)
        mol = add_bonds(mol)
    except (ValueError, IndexError):
        return False

    if mol.GetNumBonds() == 0:
        return False

    return True
