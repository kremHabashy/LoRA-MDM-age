
import torch

def compute_intra_class_variability(joint_positions, labels):
    '''
    joint_positions: Tensor of shape (N, T, J, 3)
    labels: list or tensor of class labels of length N
    Returns: dict mapping class label to average variance of joint trajectories
    '''
    labels = torch.tensor(labels)
    classes = torch.unique(labels)
    class_variability = {}

    for cls in classes:
        cls_positions = joint_positions[labels == cls]  # (Nc, T, J, 3)
        if cls_positions.size(0) == 0:
            continue
        var = cls_positions.var(dim=1)  # (Nc, J, 3)
        mean_var = var.mean(dim=(0, 2))  # (J,)
        class_variability[int(cls.item())] = mean_var.tolist()

    return class_variability
