"""Ask the weights a what-if: roll out a hand-built scene the model never saw.

Builds a synthetic seed window (constant-velocity history) from a scene spec,
then lets the trained network predict the future. No simulator involved —
this is the learned engine answering a counterfactual. Outputs are the
model's imagination, not measurements: there is no ground truth to compare
against.

Scenes are 2D, unit-domain, water-family (single material, no temperature,
no charge). Use a checkpoint trained on gen/mpm2d.py data.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import data as D
from .ckpt import eval_meta, load_checkpoint
from .rollout import rollout


def block(rng, n, x0, y0, w, h, vx=0.0, vy=0.0):
    """A uniformly filled box of n particles with one shared per-frame velocity."""
    pos = np.empty((n, 2), dtype=np.float32)
    pos[:, 0] = rng.uniform(x0, x0 + w, n)
    pos[:, 1] = rng.uniform(y0, y0 + h, n)
    return pos, np.array([vx, vy], dtype=np.float32)


# Hand-built scenes: each returns a list of (pos, vel) blocks in the unit
# domain. Velocities are per recorded frame (displacement per frame).
SCENES = {
    # two blocks: left one dropped, right one sliding left into it
    "two_blocks": lambda rng: [
        block(rng, 1500, x0=0.12, y0=0.45, w=0.22, h=0.22),
        block(rng, 1500, x0=0.62, y0=0.30, w=0.22, h=0.22, vx=-0.004),
    ],
    # classic dam break: a tall column released against the left wall
    "dam_break": lambda rng: [
        block(rng, 2600, x0=0.06, y0=0.06, w=0.24, h=0.62),
    ],
    # a droplet falling into a resting pool
    "droplet": lambda rng: [
        block(rng, 2200, x0=0.06, y0=0.05, w=0.88, h=0.16),
        block(rng, 500, x0=0.44, y0=0.72, w=0.12, h=0.12, vy=-0.006),
    ],
    # a thin tower that cannot stand
    "tower_fall": lambda rng: [
        block(rng, 1800, x0=0.44, y0=0.05, w=0.12, h=0.66),
    ],
    # a fast slug of water fired across a shallow pool
    "wave": lambda rng: [
        block(rng, 1800, x0=0.05, y0=0.05, w=0.90, h=0.12),
        block(rng, 900, x0=0.06, y0=0.24, w=0.16, h=0.22, vx=0.008),
    ],
}


def seed_window(blocks):
    """[C+1, N, 2] with constant per-frame velocity per block."""
    frames = []
    for k in range(D.HISTORY + 1):
        parts = [pos + k * vel for pos, vel in blocks]
        frames.append(np.concatenate(parts))
    return np.stack(frames).astype(np.float32)


def render(pred, path, size=420, every=2, fps=30):
    import imageio.v2 as imageio

    frames = []
    for f in range(0, len(pred), every):
        img = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)
        xs = pred[f][:, 0] * (size - 1)
        ys = (1.0 - pred[f][:, 1]) * (size - 1)
        for x, y in zip(xs, ys):
            draw.ellipse([x - 1.2, y - 1.2, x + 1.2, y + 1.2], fill="#7c5cbf")
        frames.append(np.asarray(img))
    imageio.mimsave(path, frames, fps=fps, loop=0)


def imagine(model, cfg, stats, meta, scene: str, steps: int, device,
            seed: int = 7) -> np.ndarray:
    """Roll a named scene forward from the weights -> positions [T, N, 2]."""
    rng = np.random.default_rng(seed)
    traj = D.Traj(pos=seed_window(SCENES[scene](rng)))
    pred, _ = rollout(model, traj, stats, meta, cfg, device, steps)
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="dataset dir (for meta/radius)")
    ap.add_argument("--scene", default="two_blocks", choices=sorted(SCENES))
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--gif", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg, stats, _ = load_checkpoint(args.ckpt, args.device)
    meta = eval_meta(args.data, cfg)
    pred = imagine(model, cfg, stats, meta, args.scene, args.steps,
                   args.device, seed=args.seed)
    os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
    render(pred, args.gif)
    print(f"[whatif] {args.scene}: {pred.shape[0]} frames from weights -> {args.gif}")


if __name__ == "__main__":
    main()
