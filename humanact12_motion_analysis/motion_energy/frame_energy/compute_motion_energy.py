
import torch

def compute_motion_energy(joint_positions, fps=20):
    '''
    Compute frame-level motion energy as the sum of squared joint velocities.
    joint_positions: Tensor of shape (N, T, J, 3)
    Returns: motion energy per frame (N, T-1) and summary stats
    '''
    dt = 1.0 / fps
    velocity = (joint_positions[:, 1:] - joint_positions[:, :-1]) / dt  # (N, T-1, J, 3)
    velocity_squared = velocity.pow(2).sum(dim=-1)  # (N, T-1, J)
    frame_energy = velocity_squared.sum(dim=-1)  # (N, T-1)

    mean_energy = frame_energy.mean().item()
    max_energy = frame_energy.max().item()
    std_energy = frame_energy.std().item()

    return {
        'frame_energy': frame_energy,
        'mean_energy': mean_energy,
        'max_energy': max_energy,
        'std_energy': std_energy
    }
