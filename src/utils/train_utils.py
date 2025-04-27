import torch
from src.utils.nbrs_utils import wrap_position, wrap_displacement


def pushforward_sample_steps(seed, step, pushforward):
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


def pushforward_preprocess(features, target_pos, unroll_steps):
    """ """
    position_input = features["position_sequence"]
    normalization_stats = features["normalization_stats"]
    boundaries = features["boundaries"]
    position_sequence_noise = features["position_sequence_noise"]
    
    #Add noise
    position_input = wrap_position((position_input + position_sequence_noise), boundaries)
    last_noise = position_sequence_noise[:, -1].unsqueeze(1)
    n_targets = target_pos.shape[1]
    target_noise = last_noise.repeat(1, n_targets, 1) 
    target_pos_noisy = wrap_position((target_pos + target_noise), boundaries)
    position_target = target_pos_noisy
    
    input_sequence_size = position_input.size(1)
    pos_input_and_target = torch.cat([position_input, position_target], dim=1)
    
    start_idx = max(0, input_sequence_size - 2 + unroll_steps)
    end_idx = min(pos_input_and_target.size(1), input_sequence_size  + unroll_steps + 1)
    
    pos_input_and_target = pos_input_and_target[:, start_idx:end_idx, :]

    current_velocity = wrap_displacement(
        pos_input_and_target[:, 1] -  pos_input_and_target[:, 0], boundaries
    )
    next_velocity = wrap_displacement(
        pos_input_and_target[:, 2] - pos_input_and_target[:, 1], boundaries
    )
    
    current_acceleration = next_velocity - current_velocity

    target_norm_v_acceleration = (
        current_acceleration - normalization_stats["v_acceleration"]["mean"]
    ) / normalization_stats["v_acceleration"]["std"]
    
    # if u_vel is True:
    #     u_vel_input = features["u_velocity"]
    #     u_vel_target = target_u_vel
    #     u_vel_input_and_target = torch.cat([u_vel_input, u_vel_target], dim=1)
    #     u_vel_input_and_target = u_vel_input_and_target[:, start_idx:end_idx, :]
        
    #     u_acceleration = u_vel_input_and_target[:, 2] - u_vel_input_and_target[:, 1]
        
    #     target_norm_u_acceleration = (u_acceleration - normalization_stats["u_acceleration"]["mean"]) / normalization_stats["u_acceleration"]["std"]
        
    #     return (target_norm_v_acceleration, target_norm_u_acceleration)
    
    return target_norm_v_acceleration


def pushforward_fn(features, target_positions, unroll_steps, forward_function):

    #Preprocess the input data
    target_acceleration = pushforward_preprocess(features, target_positions, unroll_steps)
        
    for _ in range(unroll_steps + 1):
        features["next_position"] = target_positions[:, unroll_steps]
        pred, _  = forward_function(features)
        
        next_pos = pushforward_integrate(
                    normalized_acceleration=pred,
                    position_sequence=features["position_sequence"],
                    normalization_stats=features["normalization_stats"]["v_acceleration"],
                    boundaries=features["boundaries"],
                    pbc=features["pbc"]
                )
                
        features["position_sequence"] = torch.cat(
                    [features["position_sequence"][:, 1:], next_pos[:, None, :]], dim=1
                )
        
    return pred, target_acceleration


def pushforward_integrate(normalized_acceleration, position_sequence, normalization_stats, boundaries, pbc=True):
    """
    The model produces the output in normalized space so we apply inverse normalization.

    Args:
        normalized_acceleration: Normalized acceleration tensor.
        position_sequence: Sequence of positions tensor.
        normalization_stats: Dictionary containing 'mean' and 'std' for normalization.
        boundaries: Tensor containing boundary conditions.

    Returns:
        Tensor of new positions after integration.
    """
    # Inverse normalize the acceleration
    acceleration = (
        normalized_acceleration * normalization_stats['std']
    ) + normalization_stats['mean']

    # Use an Euler integrator to go from acceleration to position, assuming dt = 1.
    most_recent_position = position_sequence[:, -1]
    if pbc:
        most_recent_velocity = wrap_displacement((most_recent_position - position_sequence[:, -2]), boundaries)
    
    # Update velocity and position
    new_velocity = most_recent_velocity + acceleration
    new_position = most_recent_position + new_velocity
    if pbc:
        new_position = wrap_position(new_position, boundaries)
        
    return new_position
