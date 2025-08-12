#!/usr/bin/env python3
# Render SMPL mesh (vertices + joints) from *_smpl_params.npz
import argparse
from pathlib import Path
import numpy as np
import torch
import smplx
import matplotlib
matplotlib.use("Agg")  # safe for headless nodes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def resolve_model_base(models_dir: str) -> str:
    p = Path(models_dir)
    return str(p.parent if p.name.lower()=="smpl" else p)

def build_smpl(model_base: str, gender: str, device: str):
    m = smplx.create(model_path=model_base, model_type="smpl",
                     gender=gender, use_pca=False, num_betas=10)
    m = m.to(device)
    m.eval()
    for p in m.parameters(): p.requires_grad = False
    return m

def equal_aspect(ax, P):
    mins = P.min(0); maxs = P.max(0); c = (mins+maxs)/2; r = (maxs-mins).max()/2
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits_file", required=True, help="*_smpl_params.npz")
    ap.add_argument("--models_dir", required=True, help=".../body_models or .../body_models/smpl")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--mode", choices=["points","surface"], default="points")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    data = np.load(args.fits_file, allow_pickle=True)
    poses72 = data["poses"]          # (T,72) = [3 global + 69 body]
    trans   = data["trans"]          # (T,3)
    betas   = data["betas"]          # (10,)
    gender  = str(data.get("gender","neutral"))
    T = poses72.shape[0]
    f = max(0, min(args.frame, T-1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl = build_smpl(resolve_model_base(args.models_dir), gender, device)

    g = torch.tensor(poses72[f,:3], dtype=torch.float32, device=device).unsqueeze(0)
    b = torch.tensor(poses72[f,3:], dtype=torch.float32, device=device).unsqueeze(0)
    t = torch.tensor(trans[f],      dtype=torch.float32, device=device).unsqueeze(0)
    bet = torch.tensor(betas,       dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        out = smpl(global_orient=g, body_pose=b, betas=bet, transl=t, pose2rot=True)
    V = out.vertices[0].cpu().numpy()   # (6890,3)
    J = out.joints[0].cpu().numpy()     # (24,3)

    # figure
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")

    if args.mode == "points":
        ax.scatter(V[:,0], V[:,1], V[:,2], s=0.2, alpha=0.6)
    else:
        faces = smpl.faces.astype(np.int32)
        tris = V[faces]  # (F,3,3)
        mesh = Poly3DCollection(tris, alpha=0.08, linewidth=0.2, edgecolor="k")
        ax.add_collection3d(mesh)
        ax.scatter(V[:,0], V[:,1], V[:,2], s=0.1, alpha=0.15)  # light points for depth

    ax.scatter(J[:,0], J[:,1], J[:,2], s=25, c="r", alpha=0.9)  # joints
    ax.set_title(f"SMPL Mesh (Frame {f})")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    equal_aspect(ax, V)

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.fits_file).parent / "mesh")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.fits_file).stem.replace("_smpl_params","")
    out_path = out_dir / f"{stem}_frame{f}_{args.mode}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"💾 Saved image: {out_path}")
    print(f"📁 Output directory: {out_dir}")

if __name__ == "__main__":
    main()
