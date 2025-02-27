from atomic_datasets import datatypes

import ase
import numpy as np


def ase_atoms_to_graph(molecule: ase.Atoms) -> datatypes.Graph:
    """Converts an ase Atoms object to a datatypes.Graph."""
    return datatypes.Graph(
        nodes=dict(
            positions=np.asarray(molecule.positions), species=molecule.numbers - 1
        ),
        edges=None,
        receivers=None,
        senders=None,
        globals=None,
        n_node=np.asarray([len(molecule.numbers)]),
        n_edge=None,
    )
