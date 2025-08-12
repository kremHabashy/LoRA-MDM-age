# tools/make_vc_splits.py
import os, random
from pathlib import Path

def main(root, out_dir):
    root = Path(root)
    out  = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # collect all npz under subject dirs
    items = []
    for subj_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        files = sorted(subj_dir.glob("*.npz"))
        if not files: continue
        items.append((subj_dir.name, [f"{subj_dir.name}/{f.name}" for f in files]))

    random.seed(0)
    random.shuffle(items)
    n = len(items)
    n_train = int(0.8*n); n_val = int(0.1*n)
    splits = {
        'train': items[:n_train],
        'val':   items[n_train:n_train+n_val],
        'test':  items[n_train+n_val:],
    }

    for split, pairs in splits.items():
        with open(out/f"{split}.txt", "w") as f:
            for _, rels in pairs:
                for r in rels: f.write(r + "\n")
    print({k: sum(len(r) for _, r in v) for k, v in splits.items()})

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../data/humanml3d")
    ap.add_argument("--out_dir", required=True, help=".../dataset/vc")
    args = ap.parse_args()
    main(args.root, args.out_dir)
