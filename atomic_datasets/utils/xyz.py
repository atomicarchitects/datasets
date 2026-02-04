from atomic_datasets import datatypes


def save_as_xyz(graph: datatypes.Graph, filepath: str, comment: str = "") -> None:
    """
    Save a molecular graph to XYZ format.
    
    Args:
        graph: A Graph object with nodes containing 'positions' and 'atom_types'
        filepath: Output file path
        comment: Optional comment for the second line (defaults to SMILES if available)
    """
    positions = graph['nodes']['positions']
    atom_types = graph['nodes']['atom_types']
    n_atoms = len(positions)
    
    if not comment and graph.get('properties') and 'smiles' in graph.get('properties', {}):
        comment = graph['properties']['smiles']
    
    with open(filepath, 'w') as f:
        f.write(f"{n_atoms}\n")
        f.write(f"{comment}\n")
        for atom, pos in zip(atom_types, positions):
            f.write(f"{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
