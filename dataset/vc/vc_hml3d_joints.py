import os, glob, numpy as np, torch
from torch.utils.data import Dataset

class VCHumanML3DJoints(Dataset):
    """
    Loads NPZs written by your export_humanml3d.py:
      keys: joints(T,22,3), fps, subject_id, trial_name, age, ...
    Returns:
      motion: (max_frames, 66) float32, padded with zeros
      length: int (<= max_frames)
      y: dict with 'age' (B,) float32 normalized to [0,1]
    """
    def __init__(self, root, split="train", max_frames=196, resample_fps=20,
                 age_min=None, age_max=None):
        super().__init__()
        self.root = root
        self.max_frames = max_frames
        self.resample_fps = resample_fps

        # collect files
        # expects structure: root/SUBJXX/*.npz
        self.files = sorted(glob.glob(os.path.join(root, "*", "*_humanml3d_22joints.npz")))
        # simple split by index (replace if you already dumped split lists)
        n = len(self.files)
        if split == "train":
            self.files = self.files[: int(0.9*n)]
        elif split == "test":
            self.files = self.files[int(0.9*n):]
        elif split == "val":
            self.files = self.files[int(0.8*n): int(0.9*n)]

        # scan ages to auto set min/max if not provided
        ages = []
        for p in self.files:
            with np.load(p, allow_pickle=True) as d:
                if "age" in d.files:
                    ages.append(float(d["age"]))
        self.age_min = float(np.min(ages)) if age_min is None else float(age_min)
        self.age_max = float(np.max(ages)) if age_max is None else float(age_max)
        self._eps = 1e-8

        self.nfeats = 22 * 3

    def __len__(self): return len(self.files)

    def _norm_age(self, a):
        return (a - self.age_min) / max(self._eps, (self.age_max - self.age_min))

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path, allow_pickle=True) as d:
            J = d["joints"].astype(np.float32)    # (T,22,3), already root-centered, canonical, 20fps
            age = float(d["age"]) if "age" in d.files else np.nan

        T = J.shape[0]
        # clip/pad to max_frames
        if T > self.max_frames:
            J = J[:self.max_frames]
            T = self.max_frames
        elif T < self.max_frames:
            pad = np.zeros((self.max_frames - T, 22, 3), dtype=np.float32)
            J = np.concatenate([J, pad], axis=0)

        motion = J.reshape(self.max_frames, -1)  # (max_frames, 66)
        age_norm = np.array(self._norm_age(age), dtype=np.float32)

        # package like other datasets do: x (motion), y (cond dict), length
        x = torch.from_numpy(motion)          # (F,66)
        y = {"age": torch.from_numpy(age_norm[None])}  # (1,) to keep batch dims simple
        length = int(T)

        return x, y, length
