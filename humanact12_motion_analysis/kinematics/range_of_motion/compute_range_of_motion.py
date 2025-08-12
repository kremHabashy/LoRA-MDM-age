
import numpy as np
import torch

def compute_range_of_motion(joint_positions):
    '''
    joint_positions: Tensor of shape (N, T, J, 3)
    Returns: min and max positions per joint
    '''
    min_pos = joint_positions.min(dim=1).values  # (N, J, 3)
    max_pos = joint_positions.max(dim=1).values  # (N, J, 3)
    range_of_motion = max_pos - min_pos
    return range_of_motion.mean(dim=0), range_of_motion.std(dim=0)
