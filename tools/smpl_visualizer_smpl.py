#!/usr/bin/env python3
# Visualize SMPL (24 joints) vs original markers (subset), with optional GIF export.

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

SUBSET = {"LFHD","RFHD","LBHD","RBHD","C7","T10","CLAV","STRN","LASI","RASI","SACR",
          "LSHO","RSHO","LELB","RELB","LWRA","LWRB","RWRA","RWRB",
          "LKNE","RKNE","LANK","RANK","LHEE","RHEE","LTOE","RTOE"}

def load_fitted(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    joints = d["joints"]  # (T,24,3)
    fps = float(d["fps"]) if "fps" in d else 100.0
    return joints, joints.shape[0], fps

def load_markers(processed_dir, subject, trial):
    p = Path(processed_dir)/subject/(trial+"_markers_positions.npz")
    d = np.load(p, allow_pickle=True)
    M = d["marker_data"]; names = [str(x) for x in d["marker_names"].tolist()]
    keep = [i for i,n in enumerate(names) if n.upper() in SUBSET]
    m = M[:, keep, :].copy()
    # interpolate NaNs per marker dim
    for k in range(m.shape[1]):
        for dim in range(3):
            v = m[:,k,dim]
            nans = np.isnan(v)
            if nans.any():
                idx = np.arange(len(v))
                v[nans] = np.interp(idx[nans], idx[~nans], v[~nans])
            m[:,k,dim] = v
    return m

def bounds(points):
    mn = points.reshape(-1,3).min(0)
    mx = points.reshape(-1,3).max(0)
    c = (mn+mx)/2.0
    r = (mx-mn).max()/2.0
    return c, r if r>0 else 1.0

def set_equal_aspect(ax, center, radius):
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits_file", required=True, help="*_smpl_params.npz")
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--trial", required=True)   # e.g., "SUBJ1 (0)"
    ap.add_argument("--frame", type=int, default=0, help="frame to preview (if not saving GIF)")
    ap.add_argument("--save_gif", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="-1 = last common frame")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gif_fps", type=int, default=20)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    joints, Tj, fps_fit = load_fitted(args.fits_file)   # (Tj,24,3)
    markers = load_markers(args.processed_dir, args.subject, args.trial)  # (Tm,K,3)
    T = min(Tj, markers.shape[0])

    if not args.save_gif:
        f = max(0, min(args.frame, T-1))
        J = joints[f]; M = markers[f]
        fig = plt.figure(figsize=(14,5))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        ax1.scatter(J[:,0],J[:,1],J[:,2],s=20)
        ax1.set_title("SMPL joints (24)"); ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

        ax2.scatter(M[:,0],M[:,1],M[:,2],s=20)
        ax2.set_title(f"Markers subset (K={M.shape[0]})"); ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

        both = np.vstack([J,M])
        ax3.scatter(J[:,0],J[:,1],J[:,2],s=20,alpha=0.8,label="SMPL")
        ax3.scatter(M[:,0],M[:,1],M[:,2],s=20,alpha=0.6,label="Markers")
        ax3.legend()
        ax3.set_title("Overlay"); ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")

        center, radius = bounds(both)
        for ax in (ax1,ax2,ax3):
            set_equal_aspect(ax, center, radius)
        plt.tight_layout(); plt.show()
        return

    # -------- GIF mode --------
    start = max(0, args.start)
    end = T-1 if args.end < 0 else min(args.end, T-1)
    frames = list(range(start, end+1, max(1, args.stride)))
    if len(frames) == 0:
        print("Nothing to animate (empty frame range).")
        return

    # Global bounds for stable camera
    both_all = np.concatenate([joints[:T], markers[:T]], axis=1)  # (T, 24+K, 3)
    center, radius = bounds(both_all)

    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    s1 = ax1.scatter([],[],[],s=20)
    s2 = ax2.scatter([],[],[],s=20)
    s3a = ax3.scatter([],[],[],s=20,alpha=0.8,label="SMPL")
    s3b = ax3.scatter([],[],[],s=20,alpha=0.6,label="Markers")
    ax3.legend()
    ax1.set_title("SMPL joints (24)"); ax2.set_title("Markers subset"); ax3.set_title("Overlay")
    for ax in (ax1,ax2,ax3):
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        set_equal_aspect(ax, center, radius)

    def update(fi):
        J = joints[fi]; M = markers[fi]
        s1._offsets3d = (J[:,0], J[:,1], J[:,2])
        s2._offsets3d = (M[:,0], M[:,1], M[:,2])
        s3a._offsets3d = (J[:,0], J[:,1], J[:,2])
        s3b._offsets3d = (M[:,0], M[:,1], M[:,2])
        fig.suptitle(f"Frame {fi}", y=0.98)
        return s1, s2, s3a, s3b

    ani = FuncAnimation(fig, update, frames=frames, interval=1000/args.gif_fps, blit=False)

    # Output path
    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.fits_file).parent / "animation")
    out_dir.mkdir(parents=True, exist_ok=True)
    trial = args.trial.replace(" ", "_").replace("(", "").replace(")", "")
    gif_path = out_dir / f"{args.subject}_{trial}_f{start}-{end}_s{args.stride}.gif"

    writer = PillowWriter(fps=args.gif_fps)
    ani.save(str(gif_path), writer=writer)
    print(f"💾 Saved GIF: {gif_path}")
    print(f"📁 Output directory: {out_dir}")

if __name__ == "__main__":
    main()
