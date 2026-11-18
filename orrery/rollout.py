"""Autoregressive rollout from a trained checkpoint + side-by-side GIF.

Left panel: ground truth. Right panel: prediction from the weights. Seeded
with the first HISTORY+1 frames of a trajectory, then the network predicts
every subsequent frame from its own output (temperature included on thermal
corpora — predicted dT integrates forward, so temp drift is visible as color
drift).

--metrics prints com-height checkpoints — the float-up verdict for settled
fluids (a Gen-1/2 defect: predicted com rises while truth stays put).
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import data as D
from .ckpt import eval_meta, load_checkpoint
from .spines import make_spine

# pred colors per material / truth desaturated
PRED_COLORS = ["#7c5cbf", "#e2892b", "#5ba7d9"]
TRUTH_COLORS = ["#9a9a9a", "#cbb089", "#a9c4d6"]


@torch.no_grad()
def rollout(model, traj, stats, meta, cfg, device, steps, regime: int = -1):
    wall_clip = cfg.get("wall_clip", 1.0)
    thermal = cfg.get("thermal", False)
    n = traj.pos.shape[1]
    chg = None
    if traj.charge is not None and cfg.get("charged", False):
        chg = torch.tensor(traj.charge, dtype=torch.float32, device=device)
    spine = make_spine(cfg, charges=chg)
    window = [torch.tensor(traj.pos[i], dtype=torch.float32, device=device)
              for i in range(D.HISTORY + 1)]
    temp_win = None
    synth_temp = traj.temp is None
    if thermal:
        if synth_temp:
            # merged-net eval on a temp-less corpus: isothermal at t_ref
            temp_win = [torch.full((n,), 0.5, device=device)
                        for _ in range(D.HISTORY + 1)]
        else:
            temp_win = [torch.tensor(traj.temp[i], dtype=torch.float32,
                                     device=device)
                        for i in range(D.HISTORY + 1)]
    mat = None
    if cfg.get("n_materials", 0):
        m = traj.mat if traj.mat is not None else np.zeros(n, np.uint8)
        mat = torch.tensor(m, dtype=torch.long, device=device)
    reg = None
    if cfg.get("n_regimes", 0):
        assert regime >= 0, "merged checkpoint: pass --regime (corpus index)"
        reg = torch.full((n,), regime, dtype=torch.long, device=device)

    am = torch.tensor(stats["acc_mean"], dtype=torch.float32, device=device)
    astd = torch.tensor(stats["acc_std"], dtype=torch.float32, device=device)
    lo, hi = meta.domain
    margin = meta.wall_margin
    dim = cfg.get("dim", 2)

    preds = [w.cpu().numpy() for w in window]
    pred_temps = ([t.cpu().numpy() for t in temp_win]
                  if thermal and not synth_temp else None)
    for _ in range(steps):
        pos_seq = torch.stack(window)
        p_now, p_prev = pos_seq[-1], pos_seq[-2]
        ei = D.build_edges(p_now, meta.radius)
        tw = torch.stack(temp_win) if thermal else None
        nf = D.node_features(pos_seq, stats, meta, device,
                             wall_clip=wall_clip, temp_seq=tw, charge=chg)
        ef = D.edge_features(p_now, ei, meta.radius)
        out = model(nf, ei, ef, mat=mat, regime=reg)
        acc = out[:, :dim] * astd + am
        if spine is not None:
            acc = acc + spine(p_now)
        p_next = 2 * p_now - p_prev + acc
        if meta.walls:
            p_next = torch.clamp(p_next, lo + margin * 0.5, hi - margin * 0.5)
        window = window[1:] + [p_next]
        preds.append(p_next.cpu().numpy())
        if thermal:
            d_t = out[:, dim] * stats["dtemp_std"] + stats["dtemp_mean"]
            t_next = torch.clamp(temp_win[-1] + d_t, 0.0, 1.0)
            temp_win = temp_win[1:] + [t_next]
            if pred_temps is not None:
                pred_temps.append(t_next.cpu().numpy())
    return np.stack(preds), (np.stack(pred_temps) if pred_temps else None)


def _temp_color(t: float):
    """0..1 -> blue..red."""
    r = int(60 + 195 * t)
    b = int(215 - 175 * t)
    return (r, 70, b)


def _draw_points(draw, pts, xoff, size, mats, temps, is_pred, dim):
    if dim == 3:
        order = np.argsort(-pts[:, 2])  # far-to-near
        pts = pts[order]
        if mats is not None:
            mats = mats[order]
        if temps is not None:
            temps = temps[order]
    for k in range(pts.shape[0]):
        x = xoff + pts[k, 0] * (size - 1)
        y = (1.0 - pts[k, 1]) * (size - 1)
        if temps is not None:
            color = _temp_color(float(temps[k]))
        elif mats is not None:
            color = (PRED_COLORS if is_pred else TRUTH_COLORS)[int(mats[k]) % 3]
        else:
            color = "#7c5cbf" if is_pred else "#888888"
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)


def render_gif(truth, pred, path, mats=None, truth_temps=None, pred_temps=None,
               dim=2, size=360, every=2, fps=30):
    import imageio.v2 as imageio

    frames = []
    n = min(len(truth), len(pred))
    for f in range(0, n, every):
        img = Image.new("RGB", (size * 2 + 8, size), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([size, 0, size + 7, size], fill="#dddddd")
        _draw_points(draw, truth[f], 0, size, mats,
                     truth_temps[f] if truth_temps is not None else None,
                     False, dim)
        _draw_points(draw, pred[f], size + 8, size, mats,
                     pred_temps[f] if pred_temps is not None else None,
                     True, dim)
        frames.append(np.asarray(img))
    imageio.mimsave(path, frames, fps=fps, loop=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--traj", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--metrics", action="store_true",
                    help="print com-height checkpoints (float-up verdict)")
    ap.add_argument("--regime", type=int, default=-1,
                    help="corpus index for merged checkpoints (training order)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg, stats, ck_step = load_checkpoint(args.ckpt, args.device,
                                                 regime=args.regime)
    meta = eval_meta(args.data, cfg)

    trajs = D.load_corpus(args.data)
    idx = args.traj if args.traj >= 0 else len(trajs) - 1
    truth = trajs[idx]
    steps = min(args.steps, truth.pos.shape[0] - D.HISTORY - 1)

    pred, pred_temps = rollout(model, truth, stats, meta, cfg, args.device,
                               steps, regime=args.regime)
    n = min(len(truth.pos), len(pred))
    mse = float(np.mean((truth.pos[:n] - pred[:n]) ** 2))
    print(f"[rollout] traj {idx}: {steps} predicted frames, "
          f"full-rollout position MSE={mse:.6f} (ckpt step {ck_step})")

    if args.metrics:
        for f in range(0, n, max(50, n // 8)):
            ty = truth.pos[f][:, 1].mean()
            py = pred[f][:, 1].mean()
            print(f"[metrics] frame {f:4d}: com_y truth={ty:.4f} pred={py:.4f} "
                  f"drift={py - ty:+.4f}")

    if args.gif:
        os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
        t_temps = truth.temp[:n] if truth.temp is not None else None
        p_temps = pred_temps
        if truth.charge is not None and t_temps is None:
            # color by charge sign (red +, blue -) via the temp colormap
            qmap = np.tile((truth.charge + 1.0) / 2.0, (n, 1))
            t_temps = p_temps = qmap
        render_gif(truth.pos, pred, args.gif, mats=truth.mat,
                   truth_temps=t_temps, pred_temps=p_temps,
                   dim=cfg.get("dim", 2))
        print(f"[rollout] wrote {args.gif}")


if __name__ == "__main__":
    main()
