<div align="center">

# ML Code for *Data-Driven Discretizations of Quasi-Lagrangian Turbulence*

<a href="https://github.com/pre-commit/pre-commit"><img alt="Python" src="https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white"></a> <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a> ![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)

</div>

## Description

This repository has the following structure:

- `./configs/` - **YAML configurations** for ML experiments.
- `./data/` - ML datasets. To run the ML experiments from the paper, download the datasets from Zenodo and unzip them inside this directory, e.g., `./data/2D_KOLM_4096_140kevery1/`.
- `./logs/` - ML logs. E.g., W&B, checkpoints, figures.
- `./neuraloperator/` - Modified version of the [neuraloperator](https://github.com/neuraloperator/neuraloperator) library as git module with modifications to enable periodic boundary conditions in [GINO](https://arxiv.org/abs/2309.00583).
- `./notebooks/` - Visualization of results. All plots of ML experimental results were generated with `./notebooks/vis_rlt.py`.
- `./scripts/` - **Launching ML experiments**. The three bash files contain all commands for training and inference (kolm1/kolm10/hit10).
- `./sph/` - **Baseline SPH** with CUDA implementation of TVF-SPH [(Adami et al., 2013)](https://www.sciencedirect.com/science/article/abs/pii/S002199911300096X). Provides following functionalities:
  - Solver validation experiments can be launched with `cd sph/cu/ && bash validate_tgv2d.sh` and `cd sph/cu/ && bash validate_tgv3d.sh`.
  - Generate the SPH baseline results with `cd sph/cu/ && bash validate_kolm2d.sh` and `cd sph/cu/ && bash validate_hit3d.sh`.
- `./src/` - **Core ML code** for learning quasi-Lagrangian fluid dynamics providing following parametrizations:
  - $[\mathbf{u} \leftrightarrow \mathbf{v}]$: Learn both physical velocity $\mathbf{u}$ and shifting velocity $\mathbf{v}$ simultaneously with one GNN.
  - $[\mathbf{u} \to \mathbf{v}]$: Learn the field evolution $\mathbf{u}$ with a neural operator, and then advect+relax particles.
  - $[\mathbf{v} \to \mathbf{u}]$: Learn the dynamics $\mathbf{v}$ with a GNN, and then use a second lightweight GNN to map $\mathbf{v}$ to $\mathbf{u}$.
- `./validate_code/` - experiments on reproducing results from LagrangeBench. This was the beginning of the codebase, but it hasn't been kept up-to-date and is here only as a reference.

## Installation

#### Pip

```bash
# clone project
git clone --recurse-submodules $GITHUB_HTTPS
# if you forgot `--recurse-submodule`, run `git submodule update --init`
cd $REPOSITORY_NAME

# create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# install requirements
pip install -r requirements.txt
# install this codebase
pip install -e .

pip install -e neuraloperator/
```

#### Dev

```bash
# install the pre-commit hooks from .pre-commit-config.yaml
pre-commit install
# update pre-commit hook versions
pre-commit autoupdate
# manually run pre-commit on all files
pre-commit run -a
```

#### Datasets

Add your datasets to `./data` using symbolic links.

```bash
ln -s /my/dataset/dir ./data/
```

## How to run

Train model with chosen experiment configuration.

```bash
# train on GPU
python src/train.py experiment=gns_kolm2d_every1 trainer=gpu
```
