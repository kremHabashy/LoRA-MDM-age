import os, json
from os.path import join as pjoin
from typing import List, Dict
import numpy as np
from torch.utils import data

class VCXYZDataset(data.Dataset):
    """
    Motion-only dataset for your exported npz files.
    Expects a split file listing relative NPZ paths (from root), one per line, e.g.:
      SUBJ01/SUBJ1_0_humanml3d_22joints.npz
    Returns:
      motion: Float32 [T, 66]  (xyz flattened)  Z-normalized
      cond:   {'y': {'age': Float32[B?], 'mask': Float32[1,1,T]}}
    """
    def __init__(self,
                 root: str,
                 split_file: str,
                 njoints: int = 22,
                 nfeats: int = 3,
                 max_motion_length: int = 200,
                 mean_std_dir: str = None):
        super().__init__()
        self.root = root
        self.njoints = njoints
        self.nfeats = nfeats
        self.max_motion_length = max_motion_length

        # read split
        with open(split_file, 'r') as f:
            self.files = [ln.strip() for ln in f if ln.strip()]
        assert len(self.files) > 0, f"Empty split: {split_file}"

        # meta dir (mean/std)
        if mean_std_dir is None:
            # default: ./dataset/vc/meta/<split>
            split = os.path.splitext(os.path.basename(split_file))[0]
            mean_std_dir = pjoin(os.path.dirname(split_file), 'meta', split)
        os.makedirs(mean_std_dir, exist_ok=True)
        self.mean_path = pjoin(mean_std_dir, 'mean.npy')
        self.std_path  = pjoin(mean_std_dir, 'std.npy')

        # compute or load mean/std
        if not (os.path.exists(self.mean_path) and os.path.exists(self.std_path)):
            self._compute_and_save_mean_std()
        self.mean = np.load(self.mean_path).astype(np.float32)  # [66]
        self.std  = np.load(self.std_path).astype(np.float32)   # [66]
        self.std[self.std < 1e-8] = 1e-8

        # optional: cache ages for fast access
        self._ages = [self._peek_age(pjoin(self.root, rel)) for rel in self.files]

    def _peek_age(self, npz_path: str) -> float:
        d = np.load(npz_path, allow_pickle=True)
        return float(d['age']) if 'age' in d.files else np.nan

    def _iter_train_candidates(self) -> List[np.ndarray]:
        for rel in self.files:
            path = pjoin(self.root, rel)
            d = np.load(path, allow_pickle=True)
            J = d['joints']  # [T,22,3]
            X = J.reshape(J.shape[0], -1).astype(np.float32)  # [T,66]
            yield X

    def _compute_and_save_mean_std(self):
        xs = []
        for X in self._iter_train_candidates():
            xs.append(X)
        Xcat = np.concatenate(xs, axis=0)  # [N,66]
        mean = Xcat.mean(axis=0)
        std  = Xcat.std(axis=0)
        np.save(self.mean_path, mean.astype(np.float32))
        np.save(self.std_path,  std.astype(np.float32))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        rel = self.files[idx]
        path = pjoin(self.root, rel)
        d = np.load(path, allow_pickle=True)
        J = d['joints'].astype(np.float32)      # [T,22,3]
        T = J.shape[0]

        # optional crop if too long
        if T > self.max_motion_length:
            s = np.random.randint(0, T - self.max_motion_length + 1)
            J = J[s:s+self.max_motion_length]
            T = J.shape[0]

        X = J.reshape(T, -1)                    # [T,66]
        X = (X - self.mean) / self.std          # z-norm

        age = float(d['age']) if 'age' in d.files else np.nan
        mask = np.ones((1, 1, T), dtype=np.float32)

        # training loop + diffusion expect motion as torch later; returning np is fine
        cond = {'y': {'age': np.array(age, dtype=np.float32),
                      'mask': mask}}
        return X, cond
