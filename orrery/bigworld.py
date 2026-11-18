"""Fill the card: run the small-scene-trained model on a world as large as
VRAM allows. Wide domain [0,W]x[0,1], many water blocks dropped at once.

The model is local and translation-invariant — it never knew the world was
1x1. This is the train-small-run-big test: 3k-particle training scenes,
N-hundred-thousand-particle world.

Renders a density-image GIF (histogram, not per-particle dots) and reports
particles, edges, sec/step, and peak VRAM.
"""

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from . import data as D
from .ckpt import eval_meta, load_checkpoint

DENSITY = 37_700  # particles per unit area at training spacing


def build_scene(rng, n_target):
    """Tile water blocks across a wide domain until ~n_target particles."""
    side = 0.22
    per_block = int(side * side * DENSITY)
    n_blocks = max(1, n_target // per_block)
    pitch = 0.30
    width = max(1.0, n_blocks * pitch + 0.2)
    parts = []
    for b in range(n_blocks):
        x0 = 0.1 + b * pitch
        y0 = 0.35 + 0.35 * ((b % 3) / 2.0)
        blk = np.empty((per_block, 2), dtype=np.float32)
        blk[:, 0] = rng.uniform(x0, x0 + side, per_block)
        blk[:, 1] = rng.uniform(y0, y0 + side, per_block)
        parts.append(blk)
    pos = np.concatenate(parts)
    return pos, width


def render_gif(frames_hist, path, fps=20):
    import imageio.v2 as imageio

    imgs = []
    for h in frames_hist:
        a = np.clip(h / max(1e-9, np.percentile(h, 99.5)), 0, 1)
        img = (255 - a * 200).astype(np.uint8)  # white bg, purple-ish density
        rgb = np.stack([img - (255 - img) // 3, img - (255 - img) // 2, img], -1)
        imgs.append(np.asarray(Image.fromarray(rgb.transpose(1, 0, 2)[::-1])))
    imageio.mimsave(path, imgs, fps=fps, loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--particles", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--edge-chunk", type=int, default=2_000_000)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--render-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg, stats, _ = load_checkpoint(args.ckpt, args.device)
    if (cfg.get("dim", 2) != 2 or not cfg.get("walls", True)
            or cfg.get("spine") not in (None, "uniform")
            or cfg.get("thermal") or cfg.get("n_materials")
            or cfg.get("charged") or cfg.get("n_regimes")):
        raise SystemExit("bigworld supports 2D single-material water "
                         "checkpoints (mpm2d corpora) only")
    meta = eval_meta(args.data, cfg)

    # --fp16 applies to the network only. Positions, the history window and the
    # integrator stay fp32: on a domain ~100 units wide, fp16 spacing is 0.0625
    # (several graph radii), so fp16 positions would quantize away per-frame
    # motion and corrupt the neighbor graph and the velocity features.
    model_dtype = torch.float16 if args.fp16 else torch.float32
    if args.fp16:
        model.half()

    rng = np.random.default_rng(args.seed)
    pos_np, width = build_scene(rng, args.particles)
    n = pos_np.shape[0]
    bounds = (0.0, width, 0.0, 1.0)
    print(f"[bigworld] {n} particles, domain {width:.1f}x1.0, "
          f"model dtype={model_dtype}, state dtype=torch.float32")

    pos = torch.tensor(pos_np, dtype=torch.float32, device=args.device)
    window = [pos.clone() for _ in range(D.HISTORY + 1)]
    am = torch.tensor(stats["acc_mean"], dtype=torch.float32, device=args.device)
    astd = torch.tensor(stats["acc_std"], dtype=torch.float32, device=args.device)
    g_vec = torch.tensor([0.0, -cfg.get("g_frame", 0.0)], dtype=torch.float32,
                         device=args.device)
    wall_clip = cfg.get("wall_clip", 1.0)
    margin = meta.wall_margin * 0.5

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hists, times, n_edges = [], [], 0
    bins = (min(1600, int(width * 120)), 120)

    with torch.no_grad():
        for step in range(args.steps):
            t0 = time.time()
            pos_seq = torch.stack(window)
            p_now, p_prev = pos_seq[-1], pos_seq[-2]
            ei = D.build_edges_grid(p_now, meta.radius)
            nf = D.node_features(pos_seq, stats, meta, args.device, bounds=bounds,
                                 wall_clip=wall_clip)
            ef = D.edge_features(p_now, ei, meta.radius).to(model_dtype)
            acc = model(nf.to(model_dtype), ei, ef, edge_chunk=args.edge_chunk)
            acc = acc.float() * astd + am + g_vec
            p_next = 2 * p_now - p_prev + acc
            p_next[:, 0] = torch.clamp(p_next[:, 0], 0 + margin, width - margin)
            p_next[:, 1] = torch.clamp(p_next[:, 1], 0 + margin, 1 - margin)
            window = window[1:] + [p_next]
            times.append(time.time() - t0)
            n_edges = ei.shape[1]
            if args.gif and step % args.render_every == 0:
                p = p_next.cpu().numpy()
                h, _, _ = np.histogram2d(
                    p[:, 0], p[:, 1], bins=bins, range=[[0, width], [0, 1]]
                )
                hists.append(h)
            if not torch.isfinite(p_next).all():
                print(f"[bigworld] NaN at step {step} — aborting")
                break

    peak = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else 0
    print(f"[bigworld] {n} particles | {n_edges} edges | "
          f"{np.mean(times):.3f}s/step ({1/np.mean(times):.1f} fps) | "
          f"peak VRAM {peak:.2f} GiB")
    if args.gif and hists:
        os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
        render_gif(hists, args.gif)
        print(f"[bigworld] wrote {args.gif}")


if __name__ == "__main__":
    main()
