__version__ = "0.1.4"

from .datatypes import Graph, MolecularDataset

# For legacy reasons, we allow importing datasets from the top-level namespace, but we recommend using the submodules for clarity
from .datasets import *
