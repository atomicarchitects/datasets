import os
from typing import Dict, Iterable, Optional
import numpy as np
import torch
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from rdkit.Chem.rdchem import Mol
import tqdm
import random

from atomic_datasets import datatypes
from atomic_datasets import utils


class ChemBL3DStereo(datatypes.MolecularDataset):
    """Dataset of ChemBL 3D structures."""

    def __init__(
        self,
        root_dir: str,
        split: str,
        start_index: int = None,
        end_index: int = None,
        train_on_single_molecule: Optional[bool] = False,
        train_on_single_molecule_index: Optional[int] = 0,
        # use_cache: bool = True,  # New parameter to enable/disable caching
        # cache_dir: Optional[str] = None,  # New parameter to specify cache directory
    ):
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.all_graphs = []
        self.preprocessed = False

        self.start_index = start_index
        self.end_index = end_index
        self.train_on_single_molecule = train_on_single_molecule
        self.train_on_single_molecule_index = train_on_single_molecule_index
        # self.use_cache = use_cache
        # self.cache_dir = cache_dir

        if start_index is not None:
            self.all_graphs = self.all_graphs[start_index:]

        if end_index is not None:
            self.all_graphs = self.all_graphs[:end_index]

    @classmethod
    def atom_types(cls) -> np.ndarray:
        return utils.atomic_numbers_to_symbols(cls.get_atomic_numbers())

    @classmethod
    def get_atomic_numbers(cls) -> np.ndarray:
        return np.asarray([
            1,
            5,
            6,
            7,
            8,
            9,
            13,
            14,
            15,
            16,
            17,
            33,
            35,
            53,
            80,
            83,
            34,  # nope this is not out of order
        ])
    
    @classmethod
    def species_to_atomic_numbers(cls) -> Dict[int, int]:
        return {i: atomic_number for i, atomic_number in enumerate(cls.get_atomic_numbers())}

    def preprocess(self):
        self.preprocessed = True

        if self.train_on_single_molecule:
            self.start_index = self.train_on_single_molecule_index
            self.end_index = self.train_on_single_molecule_index + 1

        # Use cached version if enabled
        # if self.use_cache:
        #     load_chembl3d_stereo_fn = utils.cache_to_file("chembl3d_stereo", self.cache_dir)(load_chembl3d_stereo)
        # else:
        load_chembl3d_stereo_fn = load_chembl3d_stereo

        all_graphs = list(
            load_chembl3d_stereo_fn(
                self.root_dir,
                self.split,
            )
        )
        # random.seed(0)
        # random.shuffle(all_graphs)
        # if start_index/end_index are None, they default to the start/end of the list when used as indices
        self.all_graphs = all_graphs[self.start_index:self.end_index]

    @utils.after_preprocess
    def __iter__(self) -> Iterable[datatypes.Graph]:
        for graph in self.all_graphs:
            yield graph

    @utils.after_preprocess
    def __len__(self) -> int:
        return len(self.all_graphs)

    @utils.after_preprocess
    def __getitem__(self, idx: int) -> datatypes.Graph:
        return self.all_graphs[idx]


def load_chembl3d_stereo(root_dir: str, split: str) -> Iterable[datatypes.Graph]:
    """Obtained from https://github.com/isayevlab/LoQI."""
    split_file = os.path.join(root_dir, "chembl3d_stereo", "processed", f"{split}_h_nomol.pt")
    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage, Mol])
    data, batch = torch.load(split_file)
    for i in tqdm.tqdm(range(len(batch["chemblid"])-1)):
        pos = data.pos[batch["pos"][i] : batch["pos"][i + 1]]
        species = data.x[batch["x"][i] : batch["x"][i + 1]]
        mol_properties = {
            "chemblid": data.chemblid[batch["chemblid"][i] : batch["chemblid"][i + 1]],
            "smiles": data.smiles[batch["smiles"][i] : batch["smiles"][i + 1]],
            "charges": data.charges[batch["charges"][i] : batch["charges"][i + 1]],
            "is_aromatic": data.is_aromatic[batch["is_aromatic"][i] : batch["is_aromatic"][i + 1]],
            "is_in_ring": data.is_in_ring[batch["is_in_ring"][i] : batch["is_in_ring"][i + 1]],
            "hybridization": data.hybridization[batch["hybridization"][i] : batch["hybridization"][i + 1]],
        }
        get_atom_types = lambda x: ChemBL3DStereo.atom_types()[x]
        frag = datatypes.Graph(
            nodes=dict(
                positions=np.asarray(pos),
                species=np.asarray(species),
                atom_types=np.asarray([get_atom_types(s) for s in species.tolist()]),
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=None,
            n_node=np.asarray([len(species)]),
            n_edge=None,
            properties=mol_properties,
        )
        yield frag
