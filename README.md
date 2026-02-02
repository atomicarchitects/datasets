# Atomic Datasets

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides an easy interface to the following 3D molecular datasets:
- QM9
- GEOM (Drugs)
- tmQM
- CATH
- Miniproteins

as well as the following toy datasets:
- Platonic Solids
- 3D Tetris Pieces

## Installation
Install directly from GitHub with [pip](https://pypi.org/project/pip/):
```bash
pip install git+https://github.com/atomicarchitects/datasets
```

## Example
```python
from atomic_datasets import QM9

# Initial preprocessing may take some time,
# but once preprocessed, read speeds should be fast!
dataset = QM9(
    root_dir="data/qm9",
    split="train",
)

for graph in dataset:
    positions = graph["nodes"]["positions"]  # (N, 3) array of 3D coordinates
    species = graph["nodes"]["species"]      # (N,) array of atomic species indices
    atom_types = graph["nodes"]["atom_types"] # (N,) array of element symbols
    properties = graph["properties"]          # dict of molecular properties
    
    print(f"Molecule: {properties['smiles']}")
    print(f"  Atoms: {atom_types}")
    print(f"  HOMO-LUMO gap: {properties['gap']:.4f} Ha")
```

### Data Format

Each sample is a dictionary with the following structure:

```python
{
    "nodes": {
        "positions": np.ndarray,   # (N, 3) atomic coordinates
        "species": np.ndarray,     # (N,) integer species indices
        "atom_types": np.ndarray,  # (N,) element symbols (e.g., 'C', 'H', 'O')
    },
    "edges": ...,
    "senders": ...,
    "receivers": ...,
    "globals": ...,
    "n_node": np.ndarray,          # number of nodes per graph
    "n_edge": ...,
    "properties": {...},           # dataset-specific properties
}
```

Available properties vary by dataset. For QM9:
- `mu`, `alpha`, `homo`, `lumo`, `gap`, `r2`, `zpve`
- `u0`, `u298`, `h298`, `g298`, `cv`
- `smiles`, `mol_id`

## License

The code in this repository is released under the MIT License.

**Note:** The datasets themselves belong to their original authors and are subject to their respective licenses and terms of use. Please refer to the original sources and cite the appropriate papers (see below) when using these datasets in your work.

## Citation

If you use this repository, please cite:
```bibtex
@software{atomic_datasets,
	title        = {{atomic\_datasets}},
	author       = {Daigavane, Ameya and Kim, Song},
	year         = 2025,
	month        = may,
	url          = {https://github.com/atomicarchitects/datasets},
	version      = {1.0.0}
}
```

<details>
<summary><strong>Dataset-specific citations</strong> (click to expand)</summary>

### QM9
```bibtex
@article{qm9,
	author = {Ramakrishnan, Raghunathan and Dral, Pavlo O. and Rupp, Matthias and von Lilienfeld, O. Anatole},
	journal = {Scientific Data},
	number = {1},
	pages = {140022},
	title = {Quantum chemistry structures and properties of 134 kilo molecules},
	volume = {1},
	year = {2014}
}
```

### GEOM
```bibtex
@article{geom,
	author = {Axelrod, Simon and G{\'o}mez-Bombarelli, Rafael},
	journal = {Scientific Data},
	number = {1},
	pages = {185},
	title = {GEOM, energy-annotated molecular conformations for property prediction and molecular generation},
	volume = {9},
	year = {2022}
}
```

### tmQM
```bibtex
@article{tmQM,
	author = {Balcells, David and Skjelstad, Bastian Bjerkem},
	journal = {Journal of Chemical Information and Modeling},
	month = {12},
	number = {12},
	pages = {6135--6146},
	title = {tmQM Dataset---Quantum Geometries and Properties of 86k Transition Metal Complexes},
	volume = {60},
	year = {2020}
}
```

### CATH
```bibtex
@article{cath2021,
	author = {Sillitoe, Ian and Bordin, Nicola and Dawson, Natalie and Waman, Vaishali P and Ashford, Paul and Scholes, Harry M and Pang, Camilla S M and Woodridge, Laurel and Rauer, Clemens and Sen, Neeladri and Abbasian, Mahnaz and Le Cornu, Sean and Lam, Su Datt and Berka, Karel and Varekova, Ivana Hutařová and Svobodova, Radka and Lees, Jon and Orengo, Christine A},
	title = {CATH: increased structural coverage of functional space},
	journal = {Nucleic Acids Research},
	volume = {49},
	number = {D1},
	pages = {D266--D273},
	year = {2021},
	doi = {10.1093/nar/gkaa1079}
}

@article{cath2024,
	author = {Waman, Vaishali P and Bordin, Nicola and Alcraft, Rachel and Vickerstaff, Robert and Rauer, Clemens and Chan, Qian and Sillitoe, Ian and Yamamori, Hazuki and Orengo, Christine},
	title = {CATH 2024: CATH-AlphaFlow Doubles the Number of Structures in CATH and Reveals Nearly 200 New Folds},
	journal = {Journal of Molecular Biology},
	volume = {436},
	number = {17},
	pages = {168551},
	year = {2024},
	doi = {10.1016/j.jmb.2024.168551}
}

@article{cath1997,
	author = {Orengo, C A and Michie, A D and Jones, S and Jones, D T and Swindells, M B and Thornton, J M},
	title = {CATH--a hierarchic classification of protein domain structures},
	journal = {Structure},
	volume = {5},
	number = {8},
	pages = {1093--1108},
	year = {1997},
	doi = {10.1016/s0969-2126(97)00260-8}
}

@article{gene3d,
	author = {Lewis, Tony E and Sillitoe, Ian and Dawson, Natalie and Lam, Su Datt and Clarke, Tristan and Lee, David and Orengo, Christine and Lees, Jonathan},
	title = {Gene3D: Extensive prediction of globular domains in proteins},
	journal = {Nucleic Acids Research},
	volume = {46},
	number = {D1},
	pages = {D1282},
	year = {2018},
	doi = {10.1093/nar/gkx1187}
}
```

### Miniproteins
```bibtex
@article{miniproteins,
	author = {Cao, Longxing and Coventry, Brian and Goreshnik, Inna and Huang, Buwei and Sheffler, William and Park, Joon Sung and Jude, Kevin M. and Markovi{\'c}, Iva and Kadam, Rameshwar U. and Verschueren, Koen H. G. and Verstraete, Kenneth and Walsh, Scott Thomas Russell and Bennett, Nathaniel and Phal, Ashish and Yang, Aerin and Kozodoy, Lisa and DeWitt, Michelle and Picton, Lora and Miller, Lauren and Strauch, Eva-Maria and DeBouver, Nicholas D. and Pires, Allison and Bera, Asim K. and Halabiya, Samer and Hammerson, Bradley and Yang, Wei and Bernard, Steffen and Stewart, Lance and Wilson, Ian A. and Ruohola-Baker, Hannele and Schlessinger, Joseph and Lee, Sangwon and Savvides, Savvas N. and Garcia, K. Christopher and Baker, David},
	journal = {Nature},
	number = {7910},
	pages = {551--560},
	title = {Design of protein-binding proteins from the target structure alone},
	volume = {605},
	year = {2022}
}
```

### 3D Tetris
```bibtex
@phdthesis{tetris,
	author = {Smidt, Tess E.},
	title = {Toward the Systematic Design of Complex Materials from Structural Motifs},
	year = {2018},
	school = {University of California, Berkeley},
	url = {https://www.proquest.com/dissertations-theses/toward-systematic-design-complex-materials/docview/2137540057/se-2}
}
```

</details>

<details>
<summary><strong>Preprocessing code acknowledgments</strong> (click to expand)</summary>

Preprocessing code adapted from:

```bibtex
@misc{anderson2019cormorant,
	title = {Cormorant: Covariant Molecular Neural Networks},
	author = {Anderson, Brandon and Hy, Truong-Son and Kondor, Risi},
	year = {2019},
	eprint = {1906.04015},
	archivePrefix = {arXiv},
	primaryClass = {physics.comp-ph},
	url = {https://arxiv.org/abs/1906.04015}
}

@misc{hoogeboom2022edm,
	title = {Equivariant Diffusion for Molecule Generation in 3D},
	author = {Hoogeboom, Emiel and Garcia Satorras, Victor and Vignac, Cl{\'e}ment and Welling, Max},
	year = {2022},
	eprint = {2203.17003},
	archivePrefix = {arXiv},
	primaryClass = {cs.LG},
	url = {https://arxiv.org/abs/2203.17003}
}

@misc{morehead2024gcdiff,
	title = {Geometry-Complete Diffusion for 3D Molecule Generation and Optimization},
	author = {Morehead, Alex and Cheng, Jianlin},
	year = {2024},
	eprint = {2302.04313},
	archivePrefix = {arXiv},
	primaryClass = {cs.LG},
	url = {https://arxiv.org/abs/2302.04313}
}

@misc{nikitin2025geomrevisited,
	title = {GEOM-Drugs Revisited: Toward More Chemically Accurate Benchmarks for 3D Molecule Generation},
	author = {Nikitin, Filipp and Dunn, Ian and Koes, David Ryan and Isayev, Olexandr},
	year = {2025},
	eprint = {2505.00169},
	archivePrefix = {arXiv},
	primaryClass = {cs.LG},
	url = {https://arxiv.org/abs/2505.00169}
}

@misc{vignac2023midi,
	title = {MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation},
	author = {Vignac, Clement and Osman, Nagham and Toni, Laura and Frossard, Pascal},
	year = {2023},
	eprint = {2302.09048},
	archivePrefix = {arXiv},
	primaryClass = {cs.LG},
	url = {https://arxiv.org/abs/2302.09048}
}
```

</details>
