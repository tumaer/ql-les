# CUDA SPH Solver

## 1. Validation cases

```bash
bash validate_tgv2d.sh
sbatch validate_tgv3d.sh
```

## Baseline runs (TODO)

```bash
# All Kolm2D test trajectories
# At 64^2, 128^2, 256^2, 512^2
sbatch validate_kolm2d.sh

# All HIT test trajectories
# At 32^3, 64^3, 128^3, 256^3
sbatch validate_hit3d.sh
```
