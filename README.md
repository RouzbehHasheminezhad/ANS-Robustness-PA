# Robustness of Preferential-Attachment Graphs:</br> Shifting the Baseline

**Authors**: Rouzbeh Hasheminezhad and Ulrik Brandes

The preliminary version of the paper is available [**here**](https://doi.org/10.1007/978-3-031-21131-7_35).
## Setup 
Confirm that a [LaTeX ](https://www.latex-project.org/get/) distribution is installed, incorporating the [amssymb](https://mirror.las.iastate.edu/tex-archive/fonts/amsfonts/doc/amssymb.pdf) and [amsmath](https://mirror.las.iastate.edu/tex-archive/macros/latex/required/amsmath/amsldoc.pdf) packages.\
Clone this GitHub repository. If `conda` is not already installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#).\
The following command creates a `conda` environment that includes required dependencies.
```
conda env create -f environment.yml
```
## Experiments
Activate the corresponding `conda` environment before executing the following steps in order.
```
conda activate ANS
```
### Generating Data
The following generates the data and corresponding log files in a `results` directory.\
Note that this script uses all available CPU cores and may take few hours to complete on a personal computer. \
To ease replication, we provide [**here**](https://polybox.ethz.ch/index.php/s/zN3q3AORlctQtTq) the `results` folder obtained after this step.
```
python data.py
```
### Generating Figures
The following generates paper's figures, and saves them in `results/figs/`.
```
python figures.py
```
### Generating Tables
The following generates the paper's tables, and saves them in `results/tables/`.
```
python figures.py
```
## Citation
If you find this repository useful, please consider citing the conference or journal paper.\
The conference paper can be cited as follows, the journal paper is currently under review.
```
@inproceedings{hasheminezhad_robustness_2023,
	series = {Studies in {Computational} {Intelligence}},
	title = {Robustness of {Preferential}-{Attachment} {Graphs}: {Shifting} the {Baseline}},
	language = {en},
	booktitle = {Complex {Networks} and {Their} {Applications} {XI}},
	publisher = {Springer},
	author = {Hasheminezhad, Rouzbeh and Brandes, Ulrik},
	editor = {Cherifi, Hocine and Mantegna, Rosario Nunzio and Rocha, Luis M. and Cherifi, Chantal and Micciche, Salvatore},
	year = {2023},
	pages = {445--456},
}
```
## Contact
In case you have questions, please contact [Rouzbeh Hasheminezhad](mailto:shashemi@ethz.ch).
