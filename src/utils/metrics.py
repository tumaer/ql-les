import torch
from typing import Dict

from src.utils.nbrs_utils import wrap_displacement


def compute_metrics(
    predictions,
    targets,
    metadata,
    active_metrics,
    boundaries,
    pbc=True,
    u_vel=False,
    metric_space="diff",
    most_recent_position=None,
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

    ### Explore MSE(pos) vs MSE(vel) ###
    ### Results: MSE(pos) accumulates much harder and we cannot compare with MSE(u)
    ### Results: We choose to work with MSE(v)
    # v_p = (x1_p - x0_p)/norm_v
    # v_t = (x1_t- x0_t)/norm_v
    # mse_v = (v_p - v_t)**2
    #       = (x1_p - x0_p - x1_t + x0_t)**2/norm_v**2
    #       = ((x1_p - x1_t) - (x0_p - x0_t))**2/norm_v**2
    #       = mse_p/norm_v**2 * norm_p**2 - 2 (x1_p - x1_t)(x0_p - x0_t)/norm_v**2
    #
    # dx0 = (x0_p - x0_t)/norm_p
    # mse_p1 = (dx0)**2
    # dx1 = (x1_p - x1_t)/norm_p
    # mse_p1 = (dx1)**2
    # mse_p = dx0**2 + dx1**2
    #       = (x0_p - x0_t)**2/norm_p**2 + (x1_p - x1_t)**2/norm_p**2
    #
    # v_p = wrap_displacement((predictions[1:] - predictions[:-1]), boundaries)
    # v_t = wrap_displacement((targets[1:] - targets[:-1]), boundaries)
    # mse_v = (simulator._norm(v_p - v_t, "vv")**2).mean(dim=(1, 2))
    #
    # dx0 = wrap_displacement((predictions - targets), boundaries)
    # mse_p = (simulator._norm(dx0, "vv")**2).mean(dim=(1, 2))
    #
    # print(mse_v)
    # print(mse_p)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.plot(mse_v.detach().cpu(), label="mse_v")
    # plt.plot(mse_p.detach().cpu(), label="mse_p")
    # plt.legend()
    # # log y
    # plt.yscale("log")
    # plt.grid()
    # plt.savefig("mse_v_p.png")

    if most_recent_position is not None:  # consider very fist 'v'
        ext_predictions = torch.cat([most_recent_position.unsqueeze(0), predictions], dim=0)
        ext_targets = torch.cat([most_recent_position.unsqueeze(0), targets], dim=0)
    else:  # ignore the first 'v' for len(loss) = traj_len - 1
        ext_predictions, ext_targets = predictions, targets
    v_p = wrap_displacement((ext_predictions[1:] - ext_predictions[:-1]), boundaries)
    v_t = wrap_displacement((ext_targets[1:] - ext_targets[:-1]), boundaries)
    dv = v_p - v_t  # difference space
    if metric_space == "norm":
        dv /= torch.tensor(metadata["vel_std"], device=dv.device)
    elif metric_space == "phys":
        dv /= metadata["dt"] * metadata["write_every"]

    d_pos = v_p = wrap_displacement((predictions - targets), boundaries)
    computed_metrics = {}
    loss_ranges = [1, 5, 10, 20, 50, 100]
    for metric_name in active_metrics:
        if metric_name == "mse":
            loss = (dv**2).mean(dim=(1, 2))
            computed_metrics["mse"] = loss
            for t in loss_ranges:
                if t < predictions.shape[0]:  # Ensure valid range
                    computed_metrics[f"mse{t}"] = loss[:t]  # Mean over time range
        elif metric_name == "mae":
            loss = torch.abs(dv).mean(dim=(1, 2))
            computed_metrics["mae"] = loss
            for t in loss_ranges:
                if t < predictions.shape[0]:
                    computed_metrics[f"mae{t}"] = loss[:t]  # Mean over time range
        elif metric_name == "e_kin":
            computed_metrics["e_kin"] = compute_kinetic_energy(
                ext_predictions, ext_targets, boundaries, metadata
            )
        elif metric_name == "mse_pos":
            loss = (d_pos**2).mean(dim=(1, 2))
            computed_metrics["mse_pos"] = loss
            for t in loss_ranges:
                if t < predictions.shape[0]:
                    computed_metrics[f"mse{t}_pos"] = loss[:t]

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


def field_mse(pred, target):
    """
    Compute the mean squared error for a field.

    Args:
        pred: Predicted field tensor.
        target: Target field tensor.

    Returns:
        Mean squared error for field.
    """
    loss = (pred - target) ** 2
    loss = loss.sum()
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
    dt = metadata["dt"] * metadata["write_every"]  # Time step
    dx = metadata["dx"]  # Spatial resolution
    dim = metadata["dim"]  # Number of spatial dimensions

    # Compute velocities for predictions and targets
    # Shape after subtraction: (time-1, nodes, dim)
    velocity_pred = (
        wrap_displacement(predictions[1::stride, :, :] - predictions[:-1:stride, :, :], boundaries)
        / dt
    )  # Divide by time step
    velocity_target = (
        wrap_displacement(targets[1::stride, :, :] - targets[:-1:stride, :, :], boundaries) / dt
    )  # Divide by time step

    # Compute kinetic energy
    # Squared velocities: (time-1, nodes, dim)
    # Summing over dim gives per-node KE: (time-1, dim)
    e_kin_pred = (velocity_pred**2).sum(1) * (dx**dim)  # Multiply by volume element
    e_kin_target = (velocity_target**2).sum(1) * (dx**dim)

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
