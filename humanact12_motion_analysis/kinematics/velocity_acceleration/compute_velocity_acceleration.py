
import numpy as np
import torch

def compute_velocity_acceleration(joint_positions, fps=20):
    '''
    joint_positions: Tensor of shape (N, T, J, 3)
    Returns: mean and max velocity and acceleration per joint
    '''
    dt = 1.0 / fps
    velocity = (joint_positions[:, 1:] - joint_positions[:, :-1]) / dt  # (N, T-1, J, 3)
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt  # (N, T-2, J, 3)

    velocity_mag = velocity.norm(dim=-1)  # (N, T-1, J)
    acceleration_mag = acceleration.norm(dim=-1)  # (N, T-2, J)

    mean_vel = velocity_mag.mean(dim=(0, 1))  # (J,)
    max_vel = velocity_mag.max(dim=1).values.mean(dim=0)

    mean_acc = acceleration_mag.mean(dim=(0, 1))  # (J,)
    max_acc = acceleration_mag.max(dim=1).values.mean(dim=0)

    root_vel = velocity_mag[..., 0].mean(dim=1)  # (N,)

    return {
        'mean_velocity': mean_vel,
        'max_velocity': max_vel,
        'mean_acceleration': mean_acc,
        'max_acceleration': max_acc,
        'root_velocity': root_vel.mean()
    }
