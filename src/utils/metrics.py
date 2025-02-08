import torch
from typing import List, Dict, Optional, Union

from src.utils.nbrs_utils import wrap_displacement


def comupte_metrics(
    predictions, targets, metadata, active_metrics, boundaries, pbc=True, u_vel=False
):
    """
    Compute metrics for the given predictions and targets.

    Args:
        predictions: Predicted positions or velocities.
        targets: Ground truth positions or velocities.
        metadata: Dictionary containing metadata (e.g., dx).
        active_metrics: List of active metrics to compute.
        boundaries: Tensor containing boundary conditions.
        pbc: Whether to use periodic boundary conditions.
        u_vel: Whether to use velocity updates.

    Returns:
        Dictionary containing computed metrics.
    """
    computed_metrics = {}
    loss_ranges = [1, 5, 10, 20, 50, 100]
    for metric_name in active_metrics:
        if metric_name == "mse":
            loss = (wrap_displacement((predictions - targets), boundaries) **2)
            computed_metrics["mse"] = loss.mean(dim=(1, 2))
            for t in loss_ranges:
                if t < predictions.shape[0]:  # Ensure valid range
                    computed_metrics[f"mse{t}"] = loss[:t]  # Mean over time range
        elif metric_name == "mae":
            loss = torch.abs(wrap_displacement((predictions - targets), boundaries))
            computed_metrics["mae"] = loss.mean(dim=(1, 2))
            for t in loss_ranges:
                if t < predictions.shape[0]: 
                    computed_metrics[f"mae{t}"] = loss[:t]  # Mean over time range
        elif metric_name == "e_kin":
            computed_metrics["e_kin"] = compute_kinetic_energy(
                predictions, targets, boundaries, metadata
            )

    return computed_metrics


def particle_mse(pred, target, non_kinematic_mask):
    """
    Compute the mean squared error for particles.

    Args:
        pred: Predicted positions tensor.
        target: Target positions tensor.
        non_kinematic_mask: Mask tensor indicating non-kinematic particles.

    Returns:
        Mean squared error for non-kinematic particles.
    """
    loss = (pred - target) ** 2
    loss = loss.sum(dim=-1)
    num_non_kinematic = non_kinematic_mask.sum()
    loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
    loss = loss.sum() / num_non_kinematic
    return loss


def compute_kinetic_energy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    boundaries: torch.Tensor,
    metadata: Dict,
    stride: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Compute Kinetic Energy with periodic boundary conditions.

    Args:
        predictions: Predicted positions tensor.
        targets: Ground truth positions tensor.
        boundaries: Tensor containing boundary conditions.
        metadata: Dictionary containing metadata (e.g., dt, dx, dim).
        stride: Stride for computing velocities.

    Returns:
        Mean squared error of kinetic energy.
    """
    # Extract metadata values
    dt = metadata["dt"] * metadata["write_every"]# Time step
    dx = metadata["dx"]                            # Spatial resolution
    dim = metadata["dim"]                          # Number of spatial dimensions
    
    # Compute velocities for predictions and targets
    # Shape after subtraction: (time-1, nodes, dim)
    velocity_pred = wrap_displacement(
        predictions[1::stride, :, :] - predictions[:-1:stride, :, :], boundaries
    ) / dt  # Divide by time step
    velocity_target = wrap_displacement(
        targets[1::stride, :, :] - targets[:-1:stride, :, :], boundaries
    ) / dt  # Divide by time step

    # Compute kinetic energy
    # Squared velocities: (time-1, nodes, dim)
    # Summing over dim gives per-node KE: (time-1, dim)
    e_kin_pred =(velocity_pred**2).sum(1) * (dx**dim)  # Multiply by volume element
    e_kin_target =(velocity_target**2).sum(1) * (dx**dim)

    # Averages over time and nodes
    e_kin_pred_mean = e_kin_pred.mean()  # Average over time and nodes
    e_kin_target_mean = e_kin_target.mean()

    # Mean squared error
    mse = ((e_kin_pred - e_kin_target) ** 2).mean()
    
    return {
        "predicted": e_kin_pred_mean,
        "target": e_kin_target_mean,
        "mse": mse,
    }

