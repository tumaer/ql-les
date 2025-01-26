import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional

def push_forward_sample_steps(seed, step, pushforward):
    """
    Sample the number of unroll steps based on the current training step and the
    specified pushforward configuration.

    Args:
        seed: Random seed for reproducibility.
        step: Current training step (int).
        pushforward: Pushforward configuration containing:
            - steps: List of training steps unlocking unroll stages.
            - unrolls: List of unroll step options for each stage.
            - probs: List of relative probabilities for each unroll step.

    Returns:
        Updated random seed and sampled unroll steps.
    """
    # Set random generator with the given seed
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Convert steps to a tensor and ensure it's ordered
    steps = torch.tensor(pushforward['steps'], dtype=torch.int32)
    assert torch.all(steps[:-1] <= steps[1:]), "Steps must be in ascending order."

    # Determine which stage is currently active based on the training step
    idx = (step > steps).sum().item()

    # Get the unrolls and probabilities for the active stages
    unroll_steps = torch.tensor(pushforward['unrolls'][:idx], dtype=torch.int64)
    probs = torch.tensor(pushforward['probs'][:idx], dtype=torch.float32)
    
    # Sample unroll steps based on the probabilities
    unroll_steps_idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=rng).item()
    updated_seed = seed + 1 # Increment the seed for reproducibility
    
    return updated_seed, unroll_steps[unroll_steps_idx].item()

def integrate(normalized_acceleration, position_sequence, normalization_stats, boundaries):
    """The model produces the output in normalized space so we apply inverse normalization."""
    # Inverse normalize the acceleration
    acceleration = (
        normalized_acceleration * normalization_stats['std']
    ) + normalization_stats['mean']

    # Use an Euler integrator to go from acceleration to position, assuming dt = 1.
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = (most_recent_position - position_sequence[:, -2])
    most_recent_velocity = wrap_displacement(most_recent_velocity, boundaries)
    
    # Update velocity and position
    new_velocity = most_recent_velocity + acceleration
    new_position = most_recent_position + new_velocity
    new_position = wrap_position(most_recent_position, boundaries)
        
    return new_position

def wrap_displacement(displacement, boundaries):
    """Wrap displacement to account for periodic boundary conditions."""
    #floating point percision messes up the results
    return (displacement + 0.5 * boundaries) % boundaries - 0.5 * boundaries

def wrap_position(position, boundaries):
    """Wrap position to account for periodic boundary conditions."""
    return position % boundaries

def particle_mse(pred, target, non_kinematic_mask):
    loss = (pred - target) ** 2
    loss = loss.sum(dim=-1)
    num_non_kinematic = non_kinematic_mask.sum()
    loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss))
    loss = loss.sum() / num_non_kinematic
    return loss

def eval_rollout(batch, simulator, metadata, num_rollout_steps, device, active_metrics, pbc=True, u_vel=False, **kwargs):
    """
    Evaluate rollout by computing the mean squared error over trajectories.

    Args:
        batch: Data batch containing input features and target positions.
        simulator: The simulation model.
        metadata: Dictionary containing metadata (e.g., boundaries).
        num_rollout_steps: Number of rollout steps to evaluate.
        device: Device to perform computations on.
        pbc: Whether to use periodic boundary conditions.

    Returns:
        Mean squared error metrics from the rollout evaluation.
    """
    boundaries = torch.tensor(metadata["bounds"], device=device)[:, 1] - torch.tensor(metadata["bounds"], device=device)[:, 0]
    
    simulator.eval()
    with torch.no_grad():
        features = {
            "enc_pos": batch.enc_pos,
            "n_particles_per_trajectory": batch.n_particles_per_trajectory,
            "particle_type": batch.particle_type,
            "target_pos": batch.target_pos,
            "bounds": boundaries
        }

        if u_vel:
            features.update({
                "u_velocity": batch.enc_u,
                "next_u_velocity": batch.target_u,
                "vel_solver": kwargs["vel_solver"]
            })

        computed_metrics = eval_single_rollout(simulator, features, num_rollout_steps, pbc, metadata, active_metrics, u_vel=u_vel)

    simulator.train()
    return computed_metrics


def eval_single_rollout(simulator, features, num_rollout_steps, pbc, metadata, active_metrics, u_vel=False):
    """
    Evaluate a single trajectory rollout.

    Args:
        simulator: The simulation model.
        features: Dictionary containing input features.
        num_rollout_steps: Number of rollout steps to evaluate.
        pbc: Whether to use periodic boundary conditions.

    Returns:
        Dictionary containing predicted and ground truth losses.
    """
    ground_truth_positions = features["target_pos"]
    current_positions = features["enc_pos"] #initial positions
    dim = current_positions.shape[-1]
    position_predictions = []
    
    if u_vel is False:
        for step in range(num_rollout_steps):
            next_position = simulator.predict_positions(
                current_positions=current_positions,
                n_particles_per_trajectory=features["n_particles_per_trajectory"],
                particle_types=features["particle_type"],
                pbc=pbc,
            )
            kinematic_mask = (features["particle_type"] == 3).bool()[:, None].expand(-1, dim)
            next_position_ground_truth = ground_truth_positions[:, step]
            next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)

            position_predictions.append(next_position)
            current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)

        position_predictions = torch.stack(position_predictions)  # (time, n_nodes, dim)
        ground_truth_positions = ground_truth_positions.permute(1, 0, 2)
        
        computed_metrics = comupte_metrics(position_predictions, ground_truth_positions, metadata, active_metrics, features["bounds"], pbc=pbc)
                
        return computed_metrics
    else:   
        current_u_velocity = features["u_velocity"] #initial u_velocity
        ground_truth_u_velocity = features["next_u_velocity"]
        vel_solver = features["vel_solver"]
        u_vel_predictions = []
        for step in range(num_rollout_steps):
            next_position, new_u_velocity = simulator.predict_positions(
                    current_positions=current_positions,
                    n_particles_per_trajectory=features["n_particles_per_trajectory"],
                    particle_types=features["particle_type"],
                    pbc=pbc,
                    u_velocity=current_u_velocity,
                    vel_solver=vel_solver
            )
            kinematic_mask = (features["particle_type"] == 3).bool()[:, None].expand(-1, dim)
            next_position_ground_truth = ground_truth_positions[:, step]
            next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)

            position_predictions.append(next_position)
            u_vel_predictions.append(new_u_velocity)

            current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)
            current_u_velocity = torch.cat([current_u_velocity[:, 1:], new_u_velocity[:, None, :]], dim=1)

        position_predictions = torch.stack(position_predictions)  # (time, n_nodes, dim)
        u_vel_predictions = torch.stack(u_vel_predictions)  # (time, n_nodes, dim)
        ground_truth_positions = ground_truth_positions.permute(1, 0, 2)
        ground_truth_u_velocity = ground_truth_u_velocity.permute(1, 0, 2)

        computed_position_metrics = comupte_metrics(position_predictions, ground_truth_positions, metadata, active_metrics, features["bounds"], pbc=pbc, u_vel=True)
        #TODO: discuss metrics for u_vel
        computed_vel_metrics = ((u_vel_predictions - ground_truth_u_velocity) ** 2).mean(dim=(1, 2))
        return (computed_position_metrics, computed_vel_metrics)

def comupte_metrics(predictions, targets, metadata, active_metrics, boundaries, pbc=True, u_vel=False):
    """
    Compute metrics for the given predictions and targets.

    Args:
        predictions: Predicted positions or velocities.
        targets: Ground truth positions or velocities.
        metadata: Dictionary containing metadata (e.g., dx).
        active_metrics: List of active metrics to compute.
        pbc: Whether to use periodic boundary conditions.

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
            computed_metrics["e_kin"] = compute_kinetic_energy(predictions, targets, boundaries, metadata, stride=10)

    return computed_metrics


def compute_kinetic_energy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    boundaries: torch.Tensor,
    metadata: Dict,
    stride: int = 10,
) -> Dict[str, torch.Tensor]:
    """Compute Kinetic Energy with periodic boundary conditions."""
    # Extract metadata values
    dt = metadata["dt"] * metadata["write_every"]  # Time step
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
    # Summing over dim gives per-node KE: (time-1, nodes)
    e_kin_pred =(velocity_pred**2).sum(1) * (dx**dim)  # Multiply by volume element
    e_kin_target =(velocity_target**2).sum(1) * (dx**dim)

    # Averages over time and nodes
    e_kin_pred_mean = e_kin_pred.mean()  # Average over time and nodes
    e_kin_target_mean = e_kin_target.mean()

    # Mean squared error
    mse = ((e_kin_pred - e_kin_target) ** 2).mean()
    return mse
    #TODO: check which metrics are more informative
    # return {
    #     "predicted": e_kin_pred_mean,
    #     "target": e_kin_target_mean,
    #     "mse": mse,
    # }

