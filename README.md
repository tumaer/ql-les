<div align="center">

# Learning Quasi-Lagrangian Turbulence

<a href="https://github.com/pre-commit/pre-commit"><img alt="Python" src="https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white"></a> <a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a> ![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)

</div>

## Description

What it does

## Installation

#### Pip

```bash
# clone project
git clone --recurse-submodules https://github.com/arturtoshev/sph_les
cd sph_les

# create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# install requirements
pip install -r requirements.txt
# install this codebase
pip install -e .
```

#### Dev

```bash
# first make sure pre-commit is installed
# then install the pre-commit hooks .pre-commit-config.yaml
pre-commit install
# update pre-commit hook versions
pre-commit autoupdate
# manually run pre-commit on all files
pre-commit run -a
```

#### Datasets

Add your datasets to `./data` using symbolic links.

```bash
ln -s /my/dataset/dir ./data/`
```

## How to run

Train model with chosen experiment configuration.

```bash
# train on GPU
python src/train.py experiment=gns_kolm2d_every1 trainer=gpu
```

## TODOs

- multiruns: with `python train.py -m seed=1,2,3,4,5` https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run
- sweeps: `python train.py -m hparams_search=mnist_optuna experiment=example`
- 6 sequential runs: `python train.py -m data.batch_size=32,64,128 model.lr=0.001,0.0005`
- Slurm: `override /hydra/launcher@_here_`
