"""Bake a what-if rollout into a compact binary for replay in another client.

Rolls out one of whatif.SCENES with a trained checkpoint and writes:

  <out>/pos.bin      float32 little-endian, shape [frames, particles, dim],
                     positions in the unit domain (clipped to [0, 1])
  <out>/meta.json    frames, particles, dim, fps, source checkpoint step
  <out>/preview.gif  a quick look for humans

The layout is deliberately trivial so a game engine or a web viewer can
bulk-decode it without a numpy dependency. Outputs are labeled as model
predictions ("dreams"), not simulation ground truth.

  python -m orrery.export --ckpt runs/gen2/latest.pt --data data/water2d \\
      --scene dam_break --steps 240 --out out/dam_break
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from .ckpt import eval_meta, load_checkpoint
from .whatif import SCENES, imagine, render


def bake(pred: np.ndarray, out_dir: str, scene: str, step: int,
         fps: int = 30, max_particles: int = 3000) -> dict:
    """Write [T,N,D] unit-domain positions as float32 LE + a meta.json."""
    os.makedirs(out_dir, exist_ok=True)
    t, n, dim = pred.shape
    keep = min(n, max_particles)
    if keep < n:
        sel = np.random.default_rng(0).choice(n, keep, replace=False)
        pred = pred[:, sel]
    q = np.clip(pred, 0.0, 1.0).astype("<f4")
    q.tofile(os.path.join(out_dir, "pos.bin"))
    meta = {
        "scene": scene,
        "frames": int(t),
        "particles": int(keep),
        "dim": int(dim),
        "fps": fps,
        "format": "f32le",
        "ckpt_step": int(step),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "model prediction, not simulation ground truth",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="dataset dir (for meta/radius)")
    ap.add_argument("--scene", required=True, choices=sorted(SCENES))
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg, stats, step = load_checkpoint(args.ckpt, args.device)
    meta = eval_meta(args.data, cfg)
    pred = imagine(model, cfg, stats, meta, args.scene, args.steps,
                   args.device, seed=args.seed)
    baked = bake(pred, args.out, args.scene, step)
    render(pred, os.path.join(args.out, "preview.gif"))
    print(f"[export] {baked['frames']} frames x {baked['particles']} particles "
          f"({baked['scene']}, ckpt step {baked['ckpt_step']}) -> {args.out}")


if __name__ == "__main__":
    main()
