<div align="center">

# Learning Quasi-Lagrangian Turbulence

<a href="https://github.com/pre-commit/pre-commit"><img alt="Python" src="https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white"></a> <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a> ![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)

</div>

## Description

We learn quasi-Lagrangian fluid dynamics using the following parametrizations:

- "$\mathbf{u} \land \mathbf{v}$": Learn both physical velocity $\mathbf{u}$ and shifting velocity $\mathbf{v}$ simultaneously with one GNN.
- "$\mathbf{u} \to \mathbf{v}$": Learn the field evolution $\mathbf{u}$ with a neural operator, and then simply advect particles.
- "$\mathbf{v} \to \mathbf{u}$": Learn the dynamics $\mathbf{v}$ with a GNN, and then use a second GNN to approximate $\mathbf{u}$.

## Installation

#### Pip

```bash
# clone project
git clone --recurse-submodules https://github.com/arturtoshev/sph_les
# if you forgot `--recurse-submodule`, run `git submodule update --init`
cd sph_les

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
