# #!/usr/bin/env python3
# """
# make_repo_manifest.py
# Create a lightweight repo manifest (sizes, trees, biggest files, git/env info)
# without relying on shell tools like `tree` or `du`.

# Usage:
#   python /u1/khabashy/LoRA-MDM/tools/make_repo_manifest.py \
#     --root /u1/khabashy/LoRA-MDM \
#     --out-subdir _manifests \
#     --max-depth 2 \
#     --data-depth 3 \
#     --top-n 200
# """

# import argparse
# import os
# import sys
# import json
# import heapq
# import time
# import subprocess
# from pathlib import Path
# from typing import Dict, List, Tuple, Iterable, Optional

# # -------------------- Helpers --------------------

# BIN_EXTS = {
#     ".npz",".npy",".pt",".pth",".pkl",".ckpt",
#     ".mp4",".gif",".zip",".tar",".tar.gz",".pdf",".c3d",".npz","._zst"
# }

# EXCLUDE_DIRS_CODE = {".git","__pycache__",".mypy_cache",".pytest_cache","wandb","logs"}
# EXCLUDE_DIRS_DATA = set()  # keep data by default; can skip with flag

# def human_bytes(n: int) -> str:
#     units = ["B","KB","MB","GB","TB","PB"]
#     f = float(n)
#     for u in units:
#         if f < 1024.0:
#             return f"{f:.1f} {u}"
#         f /= 1024.0
#     return f"{f:.1f} EB"

# def rel_depth(path: Path, base: Path) -> int:
#     try:
#         return len(path.relative_to(base).parts)
#     except Exception:
#         return 0

# def safe_write_text(p: Path, text: str):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     p.write_text(text, encoding="utf-8")

# def safe_write_tsv(p: Path, rows: Iterable[Tuple]):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     with p.open("w", encoding="utf-8") as f:
#         for r in rows:
#             f.write("\t".join(str(x) for x in r) + "\n")

# def list_files(root: Path) -> Iterable[Path]:
#     for dirpath, dirnames, filenames in os.walk(root):
#         for name in filenames:
#             yield Path(dirpath) / name

# # -------------------- Trees --------------------

# def make_tree(
#     root: Path,
#     max_depth: int = 3,
#     exclude_dirs: Optional[set] = None,
#     exclude_exts: Optional[set] = None,
# ) -> str:
#     exclude_dirs = exclude_dirs or set()
#     exclude_exts = exclude_exts or set()

#     # Build an in-memory tree of directories/files
#     Tree = Dict[str, dict]
#     def insert(tree: Tree, parts: List[str], is_file: bool):
#         node = tree
#         for i, part in enumerate(parts):
#             last = (i == len(parts) - 1)
#             if last and is_file:
#                 node.setdefault("__files__", []).append(part)
#             else:
#                 node = node.setdefault(part, {})
#     tree: Tree = {}

#     for dirpath, dirnames, filenames in os.walk(root):
#         rel = Path(dirpath).relative_to(root)
#         depth = 0 if str(rel) == "." else len(rel.parts)
#         # prune depth
#         if depth >= max_depth:
#             dirnames[:] = []  # stop deeper traversal
#         # prune dirs
#         dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith(".ipynb_checkpoints")]
#         # add files (filter by ext)
#         parts = [] if str(rel) == "." else list(rel.parts)
#         for fn in filenames:
#             if any(fn.endswith(ext) for ext in exclude_exts):
#                 continue
#             insert(tree, parts + [fn], is_file=True)
#         # ensure directories represented
#         for d in dirnames:
#             insert(tree, (parts + [d]), is_file=False)

#     # Pretty print tree
#     lines: List[str] = [str(root)]
#     def render(node: dict, prefix: str = ""):
#         # dirs first (alphabetical), files next
#         items = sorted([(k, v) for k, v in node.items() if k != "__files__"], key=lambda x: x[0])
#         files = sorted(node.get("__files__", []))
#         for i, (dirname, child) in enumerate(items):
#             last = (i == len(items) - 1 and not files)
#             tee = "└── " if last else "├── "
#             lines.append(prefix + tee + dirname + "/")
#             render(child, prefix + ("    " if last else "│   "))
#         for j, fn in enumerate(files):
#             last = (j == len(files) - 1)
#             tee = "└── " if last else "├── "
#             lines.append(prefix + tee + fn)

#     render(tree)
#     return "\n".join(lines) + "\n"

# # -------------------- Sizes --------------------

# def dir_size_index(root: Path, max_depth: int = 2, skip_dirs: Optional[set] = None) -> Dict[Path, int]:
#     """
#     Single pass walk; accumulate sizes to each ancestor bucket up to max_depth.
#     """
#     skip_dirs = skip_dirs or set()
#     size_map: Dict[Path, int] = {}
#     root_parts = len(root.parts)
#     for dirpath, dirnames, filenames in os.walk(root):
#         # prune
#         dirnames[:] = [d for d in dirnames if d not in skip_dirs]
#         # sum this dir's files
#         total = 0
#         for fn in filenames:
#             fp = Path(dirpath) / fn
#             try:
#                 total += fp.stat().st_size
#             except Exception:
#                 pass
#         # accumulate to ancestors (limited depth)
#         p = Path(dirpath)
#         while True:
#             depth = len(p.parts) - root_parts
#             if depth > max_depth:  # only bucket at levels up to max_depth
#                 p = p.parent
#                 continue
#             size_map[p] = size_map.get(p, 0) + total
#             if p == root:
#                 break
#             p = p.parent
#     return size_map

# def format_sizes(size_map: Dict[Path, int], root: Path, max_depth: int) -> str:
#     rows: List[Tuple[int, str, str]] = []
#     for p, sz in size_map.items():
#         depth = rel_depth(p, root)
#         if depth <= max_depth:
#             rows.append((depth, str(p), human_bytes(sz)))
#     rows.sort(key=lambda r: (r[0], r[1]))
#     lines = [f"# Directory sizes (depth ≤ {max_depth}) @ {root}"]
#     for depth, path_str, human in rows:
#         indent = "  " * depth
#         lines.append(f"{indent}{path_str}: {human}")
#     return "\n".join(lines) + "\n"

# # -------------------- Biggest files --------------------

# def top_n_files(root: Path, n: int = 200) -> List[Tuple[int, str]]:
#     heap: List[Tuple[int, str]] = []
#     for f in list_files(root):
#         try:
#             sz = f.stat().st_size
#         except Exception:
#             continue
#         if len(heap) < n:
#             heapq.heappush(heap, (sz, str(f)))
#         else:
#             heapq.heappushpop(heap, (sz, str(f)))
#     return sorted(heap, key=lambda x: -x[0])

# # -------------------- Git & Env --------------------

# def git_snapshot(root: Path) -> Dict[str, str]:
#     def run(args: List[str]) -> str:
#         try:
#             out = subprocess.run(args, cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#             return out.stdout.strip()
#         except Exception as e:
#             return f"(error: {e})"

#     snap = {}
#     # commit
#     snap["commit"] = run(["git", "rev-parse", "--short", "HEAD"])
#     # status
#     snap["status"] = run(["git", "status", "-s"])
#     # tracked
#     snap["tracked_files"] = run(["git", "ls-files"])
#     # untracked
#     snap["untracked_files"] = run(["git", "ls-files", "-o", "--exclude-standard"])
#     return snap

# def env_snapshot(env_name: Optional[str]) -> Dict[str, str]:
#     snap = {}
#     def run(args: List[str]) -> str:
#         try:
#             out = subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#             return out.stdout.strip()
#         except Exception as e:
#             return f"(error: {e})"
#     # Try conda env export
#     if env_name:
#         snap["conda_env_export"] = run(["conda", "env", "export", "-n", env_name])
#     else:
#         snap["conda_env_export"] = run(["conda", "env", "export"])
#     # pip freeze
#     snap["pip_freeze"] = run([sys.executable, "-m", "pip", "freeze"])
#     return snap

# def python_loc(files: List[Path]) -> List[Tuple[int, str]]:
#     out = []
#     for f in files:
#         try:
#             with f.open("r", encoding="utf-8", errors="ignore") as fh:
#                 lines = sum(1 for _ in fh)
#             out.append((lines, str(f)))
#         except Exception:
#             pass
#     return sorted(out, key=lambda x: -x[0])

# def gather_python_files_with_git(root: Path) -> List[Path]:
#     try:
#         out = subprocess.run(["git", "ls-files", "*.py"], cwd=root, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#         if out.stdout.strip():
#             return [root / p for p in out.stdout.strip().splitlines()]
#     except Exception:
#         pass
#     # fallback: glob all
#     return [p for p in root.rglob("*.py") if ".git" not in p.parts and "__pycache__" not in p.parts]

# # -------------------- Main --------------------

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root", type=str, required=True, help="Repo root (e.g., /u1/khabashy/LoRA-MDM)")
#     ap.add_argument("--out-subdir", type=str, default="_manifests", help="Subdir under root to write outputs")
#     ap.add_argument("--max-depth", type=int, default=2, help="Depth for size/index summary")
#     ap.add_argument("--data-depth", type=int, default=3, help="Depth for data tree")
#     ap.add_argument("--top-n", type=int, default=200, help="Top-N biggest files")
#     ap.add_argument("--env-name", type=str, default="moshpp37", help="Conda env name to snapshot (optional)")
#     ap.add_argument("--skip-data-sizes", action="store_true", help="Skip computing sizes under heavy data dirs")
#     args = ap.parse_args()

#     root = Path(args.root).resolve()
#     outdir = root / args.out_subdir
#     outdir.mkdir(parents=True, exist_ok=True)

#     t0 = time.time()
#     print(f"[manifest] Root: {root}")
#     print(f"[manifest] Writing to: {outdir}")

#     # 0) Sizes (depth-limited)
#     skip = set(EXCLUDE_DIRS_CODE)
#     if args.skip_data_sizes:
#         skip |= {"data", "body_models"}
#     size_map = dir_size_index(root, max_depth=args.max_depth, skip_dirs=skip)
#     sizes_txt = format_sizes(size_map, root, args.max_depth)
#     safe_write_text(outdir / "00_repo_sizes.txt", sizes_txt)

#     # 1) Git snapshot (if .git exists)
#     if (root / ".git").exists():
#         snap = git_snapshot(root)
#         safe_write_text(outdir / "01_git_status.txt", f"commit: {snap['commit']}\n\n{snap['status']}\n")
#         safe_write_text(outdir / "01_git_tracked_files.txt", snap["tracked_files"] + "\n")
#         safe_write_text(outdir / "01_git_untracked_files.txt", snap["untracked_files"] + "\n")

#     # 2) Code tree (exclude heavy bins), Data tree (shallow)
#     code_tree = make_tree(
#         root,
#         max_depth=args.max_depth + 2,  # show a bit more structure for code
#         exclude_dirs=EXCLUDE_DIRS_CODE | {"data","body_models","logs","wandb"},
#         exclude_exts=BIN_EXTS,
#     )
#     safe_write_text(outdir / "10_code_tree.txt", code_tree)

#     data_dir = root / "data"
#     if data_dir.exists():
#         data_tree = make_tree(
#             data_dir,
#             max_depth=args.data_depth,
#             exclude_dirs=EXCLUDE_DIRS_DATA,
#             exclude_exts=set(),  # show filenames so we can see counts
#         )
#         safe_write_text(outdir / "11_data_tree_L3.txt", data_tree)

#     # 3) Biggest files
#     big = top_n_files(root, n=args.top_n)
#     safe_write_tsv(outdir / "20_biggest_files.tsv", [(sz, path) for sz, path in big])

#     # 4) Python LOC
#     py_files = gather_python_files_with_git(root)
#     loc = python_loc(py_files)
#     safe_write_tsv(outdir / "30_py_loc.tsv", loc)

#     # 5) Env snapshot (best-effort)
#     env = env_snapshot(args.env_name)
#     safe_write_text(outdir / "40_env_moshpp37.yml", env.get("conda_env_export","") + "\n")
#     safe_write_text(outdir / "41_pip_freeze.txt", env.get("pip_freeze","") + "\n")

#     elapsed = time.time() - t0
#     summary = {
#         "root": str(root),
#         "outdir": str(outdir),
#         "elapsed_sec": round(elapsed, 2),
#         "artifacts": [
#             "00_repo_sizes.txt",
#             "01_git_status.txt",
#             "01_git_tracked_files.txt",
#             "01_git_untracked_files.txt",
#             "10_code_tree.txt",
#             "11_data_tree_L3.txt",
#             "20_biggest_files.tsv",
#             "30_py_loc.tsv",
#             "40_env_moshpp37.yml",
#             "41_pip_freeze.txt",
#         ]
#     }
#     safe_write_text(outdir / "README.json", json.dumps(summary, indent=2))
#     print(f"[manifest] Done in {elapsed:.1f}s. See {outdir}")

# if __name__ == "__main__":
#     main()

# save as tools/collect_repo_state.py and run: python tools/collect_repo_state.py
import json, os, sys, subprocess, shutil
from pathlib import Path
from datetime import datetime
import numpy as np

def run(cmd):
    try:
        return subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout.strip()
    except Exception as e:
        return f"[error running {' '.join(cmd)}] {e}"

def main():
    # Repo root = the git top-level if available, else CWD
    try:
        root = Path(run(["git","rev-parse","--show-toplevel"]))
        if not root.exists(): raise RuntimeError
    except Exception:
        root = Path.cwd()
    mfdir = root / "_manifests"
    mfdir.mkdir(parents=True, exist_ok=True)

    # 1) Environment snapshot
    env_txt = []
    env_txt.append(f"# Timestamp: {datetime.now().isoformat()}")
    env_txt.append(f"# CWD: {os.getcwd()}")
    env_txt.append("\n## git")
    env_txt.append(run(["git","rev-parse","HEAD"]))
    env_txt.append(run(["git","status","-s"]))
    env_txt.append(run(["git","remote","-v"]))

    env_txt.append("\n## conda env (moshpp37)")
    env_txt.append(run(["conda","env","export","-n","moshpp37"]))

    env_txt.append("\n## python versions")
    env_txt.append(run(["python","-V"]))
    env_txt.append(run(["python","-c","import sys;print(sys.version)"]))
    env_txt.append(run(["python","-c","import torch,platform;print('torch',torch.__version__);print('cuda',torch.version.cuda);print('is_cuda_available',torch.cuda.is_available());print('platform',platform.platform())"]))

    env_txt.append("\n## CUDA / Drivers")
    env_txt.append(run(["nvidia-smi"]))
    env_txt.append(run(["nvcc","--version"]))

    (mfdir / "12_env_snapshot.txt").write_text("\n".join(env_txt))

    # 2) HumanML3D data sanity
    data_root = root / "data" / "humanml3d"
    summary = {
        "root": str(data_root),
        "subjects": {},
        "totals": {"files":0, "frames_min":None, "frames_med":None, "frames_max":None, "with_age":0}
    }
    lengths = []
    age_count = 0
    file_count = 0
    for npz in data_root.rglob("*_humanml3d_22joints.npz"):
        try:
            d = np.load(npz, allow_pickle=True)
            J = d["joints"]
            T = int(J.shape[0])
            file_count += 1
            lengths.append(T)
            subj = str(d.get("subject_id","UNKNOWN"))
            has_age = ("age" in d.files) and (d["age"] is not None)
            if has_age: age_count += 1
            s = summary["subjects"].setdefault(subj, {"files":0, "frames":[],"with_age":0})
            s["files"] += 1
            s["frames"].append(T)
            if has_age: s["with_age"] += 1
        except Exception:
            pass

    summary["totals"]["files"] = file_count
    if lengths:
        arr = np.array(lengths)
        summary["totals"]["frames_min"] = int(arr.min())
        summary["totals"]["frames_med"] = float(np.median(arr))
        summary["totals"]["frames_max"] = int(arr.max())
    summary["totals"]["with_age"] = age_count

    # Per-subject quick stats
    for subj, s in summary["subjects"].items():
        if s["frames"]:
            arr = np.array(s["frames"])
            s["frames_min"] = int(arr.min())
            s["frames_med"] = float(np.median(arr))
            s["frames_max"] = int(arr.max())
        # shrink
        s.pop("frames", None)

    (mfdir / "13_hml3d_data_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote:\n  {mfdir/'12_env_snapshot.txt'}\n  {mfdir/'13_hml3d_data_summary.json'}")

    # 3) (optional) minimal config snapshot
    cfg_txt = []
    for p in ["train","data_loaders","utils","model","diffusion"]:
        pth = root / p
        if pth.exists():
            cfg_txt.append(f"\n## tree {p} (depth 2)")
            cfg_txt.append(run(["bash","-lc", f"cd {root} && find {p} -maxdepth 2 -type f | sort"]))
    (mfdir / "14_config_snapshot.txt").write_text("\n".join(cfg_txt))

if __name__ == "__main__":
    main()
