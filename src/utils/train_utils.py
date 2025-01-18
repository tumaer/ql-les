import torch

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
    acceleration_stats = normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats['std']
    ) + acceleration_stats['mean']

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

def eval_single_rollout(simulator, features, num_rollout_steps, device, pbc, batch_size):
    initial_positions = features["enc_pos"]
    ground_truth_positions = features["target_pos"]
    dim = initial_positions.shape[-1]
    current_positions = initial_positions
    
    predictions = []
    for step in range(num_rollout_steps):
        next_position = simulator.predict_positions(
            current_positions=initial_positions,
            n_particles_per_trajectory=features["n_particles_per_trajectory"],
            particle_types=features["particle_type"],
            pbc=pbc,
            batch_size=batch_size,
        )  # (n_nodes, dim)
        # Update kinematic particles from prescribed trajectory.
        kinematic_mask = (features["particle_type"] == 3).clone().detach()
        next_position_ground_truth = ground_truth_positions[:, step]
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, dim)
        next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)
        predictions.append(next_position)
        current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)
    predictions = torch.stack(predictions)  # (time, n_nodes, 2)
    ground_truth_positions = ground_truth_positions.permute(1, 0, 2)
    loss = (predictions - ground_truth_positions) ** 2
    output_dict = {
        "initial_positions": initial_positions.permute(1, 0, 2).cpu().numpy(),
        "predicted_rollout": predictions.cpu().numpy(),
        "ground_truth_rollout": ground_truth_positions.cpu().numpy(),
        "particle_types": features["particle_type"].cpu().numpy(),
    }
    return output_dict, loss

def eval_rollout(batch, simulator, metadata, num_rollout_steps, num_eval_steps=1, pbc=True, batch_size=1, save_results=False, device='cuda'):
    #TODO: https://github.com/jax-md/jax-md/blob/main/jax_md/space.py#L217
    eval_loss = []
    simulator.eval()
    with torch.no_grad():
        features = {
        "enc_pos": batch.enc_pos, # 
        "n_particles_per_trajectory": batch.n_particles_per_trajectory,
        "particle_type": batch.particle_type,
        "target_pos": batch.target_pos}

        trajectory_rollout, loss = eval_single_rollout(simulator, features, num_rollout_steps, device, pbc, batch_size)
        trajectory_rollout['metadata'] = metadata
        eval_loss.append(loss)
    simulator.train()
    return torch.stack(eval_loss).mean()