
import torch

def compute_foot_contact_percentage(joint_positions, left_idx=7, right_idx=8, threshold=0.01, fps=20):
    '''
    Compute percentage of frames where foot joints are in contact.
    joint_positions: Tensor of shape (N, T, J, 3)
    Returns: percentage of contact frames per sequence and overall mean
    '''
    dt = 1.0 / fps
    velocity = (joint_positions[:, 1:] - joint_positions[:, :-1]) / dt  # (N, T-1, J, 3)
    foot_vel = velocity[:, :, [left_idx, right_idx]]  # (N, T-1, 2, 3)
    foot_speed = foot_vel.norm(dim=-1)  # (N, T-1, 2)

    contact_mask = foot_speed < threshold  # (N, T-1, 2)
    contact_percentage = contact_mask.float().mean(dim=1)  # (N, 2)

    return contact_percentage.mean(dim=0), contact_percentage.std(dim=0)
