#!/bin/bash
# Validate the SPH code used for relaxations by running

# cd sph/torch

# First time to create figs_0/tgv_2d_final_tvf.pt file
# Second time to rerun starting from relaxed r
python ../../src/models/components/sph.py && python ../../src/models/components/sph.py
