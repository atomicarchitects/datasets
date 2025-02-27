# Atomic Datasets

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
or [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
uv pip install git+https://github.com/atomicarchitects/datasets
```

## Example
```python
from atomic_datasets import QM9Dataset

dataset = QM9Dataset(
    root_dir="data/qm9",
    check_with_rdkit=True,
)

for graph in dataset:
	# graph is a dictionary.
	print(graph["nodes"], graph["properties"])
```

## Citation

If you use this repository, please cite the original papers for the relevant datasets:
- QM9:
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
- GEOM:
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
Preprocessed GEOM (Drugs) data was obtained from:
```bibtex
@dataset{geom-preprocessed,
	author = {Alex Morehead and Jianlin, Cheng},
	doi = {10.5281/zenodo.7881981},
	month = apr,
	publisher = {Zenodo},
	title = {Replication Data for: EDM},
	url = {https://doi.org/10.5281/zenodo.7881981},
	version = {1.2.0},
	year = 2023,
	bdsk-url-1 = {https://doi.org/10.5281/zenodo.7881981}}
```
- tmQM:
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
- CATH:
```bibtex
@article{
    author = {
Ian Sillitoe and Nicola Bordin and Natalie Dawson and Vaishali P Waman and Paul Ashford and Harry M Scholes  and Camilla S M Pang and Laurel Woodridge and Clemens Rauer and Neeladri Sen and Mahnaz Abbasian and Sean Le Cornu and Su Datt Lam and Karel Berka and Ivana Hutařová Varekova and Radka Svobodova and Jon Lees and Christine A Orengo},
    title={CATH: increased structural coverage of functional space},
    journal={Nucleic Acids Res.},
    year={2021},
    month={1},
    day={8},
    volume={9},
    number={D1},
    pages={D266-D273},
    doi={10.1093/nar/gkaa1079},
}
@article{
    author={Vaishali P Waman and Nicola Bordin and Rachel Alcraft and Robert Vickerstaff and Clemens Rauer and Qian Chan and Ian Sillitoe and Hazuki Yamamori and Christine Orengo},
    title={CATH 2024: CATH-AlphaFlow Doubles the Number of Structures in CATH and Reveals Nearly 200 New Folds},
    journal={J Mol Biol},
    year={2024},
    month={9},
    day={1},
    volume={436},
    number={17},
    pages=168551,
    doi={10.1016/j.jmb.2024.168551},
}
@article{
    author={
C A Orengo and A D Michie and S Jones and D T Jones and M B Swindells and J M Thornton},
    title={CATH--a hierarchic classification of protein domain structures},
    journal={Structure},
    year=1997,
    month=8,
    day=15,
    volume=5,
    number=8,
    pages={1093-108},
    doi={10.1016/s0969-2126(97)00260-8}
}
@article{
    author={
        Tony E Lewis and Ian Sillitoe and Natalie Dawson and Su Datt Lam and Tristan Clarke and David Lee and Christine Orengo and Jonathan Lees
    },
    title={Gene3D: Extensive prediction of globular domains in proteins},
    journal={Nucleic Acids Res.},
    year=2018,
    month=1,
    day=4,
    volume=46,
    number={D1},
    pages={D1282},
    doi={10.1093/nar/gkx1187}
}
```
- Miniproteins:
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
- 3D Tetris:
```bibtex
@phdthesis{
    author={Smidt, Tess E.},
    year={2018},
    title={Toward the Systematic Design of Complex Materials from Structural Motifs},
    journal={ProQuest Dissertations and Theses},
    pages={200},
    note={Copyright - Database copyright ProQuest LLC; ProQuest does not claim copyright in the individual underlying works; Last updated - 2023-03-04},
    language={English},
    url={https://www.proquest.com/dissertations-theses/toward-systematic-design-complex-materials/docview/2137540057/se-2},
}
```
Preprocessing code adapted from:
```bibtex
@misc{anderson2019cormorantcovariantmolecularneural,
      title={Cormorant: Covariant Molecular Neural Networks}, 
      author={Brandon Anderson and Truong-Son Hy and Risi Kondor},
      year={2019},
      eprint={1906.04015},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/1906.04015}, 
}
@misc{hoogeboom2022equivariantdiffusionmoleculegeneration,
      title={Equivariant Diffusion for Molecule Generation in 3D}, 
      author={Emiel Hoogeboom and Victor Garcia Satorras and Clément Vignac and Max Welling},
      year={2022},
      eprint={2203.17003},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2203.17003}, 
}
@misc{morehead2024geometrycompletediffusion3dmolecule,
      title={Geometry-Complete Diffusion for 3D Molecule Generation and Optimization}, 
      author={Alex Morehead and Jianlin Cheng},
      year={2024},
      eprint={2302.04313},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2302.04313}, 
}
```


