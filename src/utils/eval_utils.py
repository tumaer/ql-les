"""Utility functions for evaluation."""
import os
import pickle
import numpy as np
import torch

from omegaconf import DictConfig
from lightning.pytorch.loggers import WandbLogger, Logger
from typing import List

from src.utils.metrics import compute_metrics


def eval_rollout(
    batch,
    simulator,
    metadata,
    num_rollout_steps,
    device, 
    active_metrics,
    pbc=True, 
    u_vel=False, 
    vis_config=None,
    trajectory_idx=0, 
    metric_space="norm",
    **kwargs
):
    """
    Evaluate rollout by computing the mean squared error over trajectories.

    Args:
        batch: Data batch containing input features and target positions.
        simulator: The simulation model.
        metadata: Dictionary containing metadata (e.g., boundaries).
        num_rollout_steps: Number of rollout steps to evaluate.
        device: Device to perform computations on.
        active_metrics: List of active metrics to compute.
        pbc: Whether to use periodic boundary conditions.
        u_vel: Whether to use velocity updates.
        out_type: Output type for saving results (e.g., 'vtk').

    Returns:
        Mean squared error metrics from the rollout evaluation.
    """
    boundaries = torch.tensor(metadata["bounds"], device=device)
    boundaries = boundaries[:, 1] - boundaries[:, 0]
    
    simulator.eval()
    with torch.no_grad():
        features = {
            "enc_pos": batch.enc_pos,
            "n_particles_per_trajectory": batch.n_particles_per_trajectory,
            "particle_types": batch.particle_types,
            "target_pos": batch.target_pos,
            "bounds": boundaries
        }
        if u_vel:
            features["u_velocity"] = batch.enc_u
            features["next_u_velocity"] = batch.target_u
            
        computed_metrics, rollout, ground_truth = eval_single_rollout(
            simulator, features, num_rollout_steps, pbc, metadata, active_metrics, u_vel=u_vel,
            metric_space=metric_space
        )

        if not u_vel:
            trajectory_rollout, ground_truth_positions = rollout, ground_truth
            kwargs_write = {}
        else:
            trajectory_rollout, u_vel_rollout = rollout
            ground_truth_positions, ground_truth_u_velocity = ground_truth
            kwargs_write = {
                "u_vel_rollout": u_vel_rollout,
                "ground_truth_u_velocity": ground_truth_u_velocity
            }

        write_rollout(
            batch=batch,
            trajectory_idx=trajectory_idx,
            trajectory_rollout=trajectory_rollout, 
            ground_truth_positions=ground_truth_positions, 
            vis_config=vis_config, 
            u_vel=u_vel,
            **kwargs_write
        )

    simulator.train()
    return computed_metrics


def eval_single_rollout(
    simulator, features, num_rollout_steps, pbc, metadata, active_metrics, u_vel=False,
    metric_space = "norm"
):
    """
    Evaluate a single trajectory rollout.

    Args:
        simulator: The simulation model.
        features: Dictionary containing input features.
        num_rollout_steps: Number of rollout steps to evaluate.
        pbc: Whether to use periodic boundary conditions.
        metadata: Dictionary containing metadata (e.g., boundaries).
        active_metrics: List of active metrics to compute.
        u_vel: Whether to use velocity updates.

    Returns:
        Dictionary containing predicted and ground truth losses.
    """
    ground_truth_positions = features["target_pos"]  # (N, T_out, D)
    current_positions = features["enc_pos"] #initial positions (N, T_in, D)
    dim = current_positions.shape[-1]
    position_predictions = []
    
    if u_vel is False:
        for step in range(num_rollout_steps):
            next_position = simulator.predict_positions(
                current_positions=current_positions,
                n_particles_per_trajectory=features["n_particles_per_trajectory"],
                particle_types=features["particle_types"],
                pbc=pbc,
            )
            kinematic_mask = (features["particle_types"] == 3).bool()[:, None].expand(-1, dim)
            next_position_ground_truth = ground_truth_positions[:, step]
            next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)

            position_predictions.append(next_position)
            current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)

        position_predictions = torch.stack(position_predictions)  # (time, n_nodes, dim)
        ground_truth_positions = ground_truth_positions.permute(1, 0, 2)
        
        trajectory_rollout = position_predictions
        
        computed_metrics = compute_metrics(
            position_predictions, ground_truth_positions, metadata, active_metrics, 
            features["bounds"], pbc=pbc, metric_space=metric_space,
            most_recent_position=features["enc_pos"][:,-1],
        )
                
        return computed_metrics, trajectory_rollout, ground_truth_positions
    else:   
        current_u_velocity = features["u_velocity"] #initial u_velocity
        ground_truth_u_velocity = features["next_u_velocity"]
        u_vel_predictions = []
        for step in range(num_rollout_steps):
            next_position, new_u_velocity = simulator.predict_positions(
                    current_positions=current_positions,
                    n_particles_per_trajectory=features["n_particles_per_trajectory"],
                    particle_types=features["particle_types"],
                    pbc=pbc,
                    u_velocity=current_u_velocity,
            )
            kinematic_mask = (features["particle_types"] == 3).bool()[:, None].expand(-1, dim)
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

        trajectory_rollout = position_predictions
        u_vel_rollout = u_vel_predictions
    
        du = u_vel_predictions - ground_truth_u_velocity  # physical space

        if metric_space == "norm":
            du /= torch.tensor(metadata["u_std"], device=du.device)
        elif metric_space == "diff":
            du *= metadata["dt"] * metadata["write_every"]

        computed_position_metrics = compute_metrics(
            position_predictions, ground_truth_positions, metadata, active_metrics, 
            features["bounds"], pbc=pbc, u_vel=True, metric_space=metric_space,
            most_recent_position=features["enc_pos"][:,-1],
        )
        # print(computed_position_metrics["mse"])
        # TODO: eval u metrics on grid!
        computed_vel_metrics = (du ** 2).mean(dim=(1, 2))
        # print(computed_vel_metrics)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # plt.plot(computed_position_metrics["mse"].detach().cpu(), label="mse_v")
        # plt.plot(computed_vel_metrics.detach().cpu(), label="mse_u")
        # plt.legend()
        # # log y
        # plt.yscale("log")
        # plt.title(f"{computed_position_metrics["mse"].mean().item():.4f}, {computed_vel_metrics.mean().item():.4f}")
        # plt.grid()
        # plt.savefig("mse_v_u.png")

        return (
            (computed_position_metrics, computed_vel_metrics),
            (trajectory_rollout, u_vel_rollout),
            (ground_truth_positions, ground_truth_u_velocity)
        )


def write_rollout(
    batch, trajectory_idx, trajectory_rollout, ground_truth_positions, vis_config, u_vel=False,
    **kwargs
) -> None:
    
    out_type = vis_config.get("out_type", None)
    rollout_dir = vis_config.get("rollout_dir", None)
    
    if rollout_dir is not None:
        os.makedirs(rollout_dir, exist_ok=True)
        
        batch_idx = trajectory_idx
        
        # Prepare the input positions for the rollout
        # (batch_size*nodes, t, dim) -> (t, batch_size*nodes, dim)
        pos_input_batch = batch.enc_pos.permute(1, 0, 2)

        batch_offset = 0  # Keeps track of the starting index for each trajectory
        for j in range(batch.batch_size):  # Write every trajectory to file (batch loop)
            num_particles = batch.n_particles_per_trajectory[j]  # Get the number of particles for this trajectory

            # Slice the pos_input_batch for the current trajectory
            pos_input = pos_input_batch[:, batch_offset:batch_offset + num_particles]  # Shape: (t, nodes, dim)
            example_rollout = trajectory_rollout[:, batch_offset:batch_offset + num_particles]  # Shape: (extrap, nodes, dim)
            
            # Prepare initial positions and the full sequence
            initial_positions = pos_input  # Shape: (t_window, nodes, dim)
            example_full = torch.concatenate([initial_positions, example_rollout], axis=0)  # Shape: (t_window + extrap, nodes, dim)

            # Collect the ground truth rollout:
            ground_truth_rollout = torch.concat([
                pos_input, ground_truth_positions[:, batch_offset:batch_offset + num_particles]
            ], axis=0)  # Shape: (t, nodes, dim)
            
            example_rollout_dict = {
                    "predicted_rollout": example_full.cpu().numpy(),  # Convert to NumPy
                    "ground_truth_rollout": ground_truth_rollout.cpu().numpy(),  # Convert to NumPy
                    "particle_types": batch.particle_types[
                        batch_offset:batch_offset + num_particles
                    ].cpu().numpy(),  # Convert to NumPy
                }
            
            if u_vel:
                u_vel_input = batch.enc_u.permute(1, 0, 2)
                u_vel_rollout = kwargs["u_vel_rollout"]
                ground_truth_u_velocity = kwargs["ground_truth_u_velocity"]
                u_vel_input = u_vel_input[:, batch_offset:batch_offset + num_particles]
                example_u_vel_rollout = u_vel_rollout[:, batch_offset:batch_offset + num_particles]
                initial_u_vel = u_vel_input
                example_u_vel_full = torch.concatenate([initial_u_vel, example_u_vel_rollout], axis=0) 
                ground_truth_u_vel_rollout = torch.concat([
                    u_vel_input, ground_truth_u_velocity[:, batch_offset:batch_offset + num_particles]
                ], axis=0)  # Shape: (t, nodes, dim)
                example_rollout_dict["predicted_u_vel"] = example_u_vel_full.cpu().numpy()
                example_rollout_dict["ground_truth_u_vel"] = ground_truth_u_vel_rollout.cpu().numpy()
    
            batch_offset += num_particles
            # File handling
            file_prefix = os.path.join(rollout_dir, f"rollout_{batch_idx * batch.batch_size + j:04d}")
            if out_type == "vtk":  # Write vtk files for each time step
                for k in range(example_full.shape[0]):
                    # Predictions
                    state_vtk = {
                        "r": example_rollout_dict["predicted_rollout"][k],
                        "tag": example_rollout_dict["particle_types"],
                    }
                    write_vtk(state_vtk, f"{file_prefix}_{k}.vtk")
                for k in range(ground_truth_rollout.shape[0]):
                    # Ground truth reference
                    ref_state_vtk = {
                        "r": example_rollout_dict["ground_truth_rollout"][k],
                        "tag": example_rollout_dict["particle_types"],
                    }
                    write_vtk(ref_state_vtk, f"{file_prefix}_ref_{k}.vtk")
            elif out_type == "pkl":
                filename = f"{file_prefix}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(example_rollout_dict, f)


def write_vtk(data_dict, path):
    """Store a .vtk file for ParaView."""

    try:
        import pyvista
    except ImportError:
        raise ImportError("Please install pyvista to write VTK files.")

    r = data_dict["r"]
    N, dim = r.shape

    # PyVista treats the position information differently than the rest
    if dim == 2:
        r = np.hstack([r, np.zeros((N, 1))])
    data_pv = pyvista.PolyData(r)

    # copy all the other information also to pyvista, using plain numpy arrays
    for k, v in data_dict.items():
        # skip r because we already considered it above
        if k == "r":
            continue

        # working in 3D or scalar features do not require special care
        if dim == 2 and v.ndim == 2:
            v = np.hstack([v, np.zeros((N, 1))])

        data_pv[k] = v

    data_pv.save(path)


def pkl2vtk(src_path, dst_path=None):
    """Convert a rollout pickle file to a set of vtk files.

    Args:
        src_path (str): Source path to .pkl file.
        dst_path (str, optoinal): Destination directory path. Defaults to None.
            If None, then the vtk files are saved in the same directory as the pkl file.

    Example:
        pkl2vtk("rollout/test/rollout_0.pkl", "rollout/test_vtk")
        will create files rollout_0_0.vtk, rollout_0_1.vtk, etc. in the directory
        "rollout/test_vtk"
    """

    # set up destination directory
    if dst_path is None:
        dst_path = os.path.dirname(src_path)
    os.makedirs(dst_path, exist_ok=True)

    # load rollout
    with open(src_path, "rb") as f:
        rollout = pickle.load(f)

    file_prefix = os.path.join(dst_path, os.path.basename(src_path).split(".")[0])
    for k in range(rollout["predicted_rollout"].shape[0]):
        # predictions
        state_vtk = {
            "r": rollout["predicted_rollout"][k],
            "tag": rollout["particle_types"],
        }
        if "predicted_u_vel" in rollout:
            state_vtk["u"] = rollout["predicted_u_vel"][k]
        write_vtk(state_vtk, f"{file_prefix}_{k}.vtk")
        # ground truth reference
        state_vtk = {
            "r": rollout["ground_truth_rollout"][k],
            "tag": rollout["particle_types"],
        }
        if "ground_truth_u_vel" in rollout:
            state_vtk["u"] = rollout["ground_truth_u_vel"][k]
        write_vtk(state_vtk, f"{file_prefix}_ref_{k}.vtk")

  
def update_wandb_id(cfg: DictConfig, logger: List[Logger]) -> None:
    """
    Retrieves the WandB run ID from the logger and updates the config paths with it.

    Args:
        cfg (DictConfig): The Hydra configuration to update.
        logger (List[Logger]): List of loggers to extract the WandB run ID.
    """

    # Retrieve the WandB run ID from the logger
    wandb_run_id = None
    for x_logger in logger:
        if isinstance(x_logger, WandbLogger):
            # wandb_run_id = x_logger.experiment.id  # Access WandB run ID
            wandb_run_id = x_logger.save_dir[-19:]  # use same date_time dir name as wandb log

    # Update the config paths if a WandB run ID is available
    if wandb_run_id:
        if cfg.model.visualize.vis_test.rollout_dir:
            cfg.model.visualize.vis_test.rollout_dir = cfg.model.visualize.vis_test.rollout_dir.replace("wandb_id", wandb_run_id)
        if cfg.model.visualize.vis_val.rollout_dir:
            cfg.model.visualize.vis_val.rollout_dir = cfg.model.visualize.vis_val.rollout_dir.replace("wandb_id", wandb_run_id)
