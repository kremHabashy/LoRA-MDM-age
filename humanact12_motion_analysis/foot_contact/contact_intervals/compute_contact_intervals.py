
import torch

def compute_contact_intervals(joint_positions, foot_idx=7, threshold=0.01, fps=20):
    '''
    Computes lengths of continuous foot contact intervals.
    joint_positions: Tensor of shape (N, T, J, 3)
    Returns: List of contact interval durations per sequence
    '''
    dt = 1.0 / fps
    velocity = (joint_positions[:, 1:] - joint_positions[:, :-1]) / dt  # (N, T-1, J, 3)
    speed = velocity[:, :, foot_idx].norm(dim=-1)  # (N, T-1)
    contact_mask = (speed < threshold)  # (N, T-1)

    intervals = []
    for seq in contact_mask:
        in_contact = seq.tolist()
        count = 0
        for val in in_contact:
            if val:
                count += 1
            elif count > 0:
                intervals.append(count)
                count = 0
        if count > 0:
            intervals.append(count)
    return intervals
