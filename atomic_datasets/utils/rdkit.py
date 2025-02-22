
from rdkit import Chem

def atom_number_to_symbol(atomic_number: int) -> str:
    """Convert atomic number to atomic symbol."""
    return Chem.PeriodicTable.GetElementSymbol(atomic_number)

def atom_symbol_to_number(atomic_symbol: str) -> int:
    """Convert atomic symbol to atomic number."""
    return Chem.PeriodicTable.GetAtomicNumber(atomic_symbol)

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
