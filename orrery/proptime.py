"""Property time (Gen 7 prototype): activity-gated rollout.

Time is a function of property (docs/design.md): no universal clock. Quiet regions
freeze — no compute, no time; active regions run at full rate. This is the
two-rung prototype:

  hot rung   particles with recent motion above threshold (+ a halo of their
             R-neighbors): the model integrates them EVERY step, on the
             subgraph induced by hot ∪ halo.
  cold rung  everything else: frozen in place (settled fluid's residual
             cancels gravity — holding still IS its physics).

Every --coarse-every steps a FULL-graph pass runs: everyone integrates,
activity reclassifies, pressure waves wake sleepers. Hysteresis keeps
particles hot for a minimum dwell so the mask doesn't flicker.

Verdict artifacts:
  - speedup: wall-clock gated vs full rollout, same ckpt, same scene
  - fidelity: position MSE of gated vs full rollout (the gating error),
    alongside each vs truth (is the gating error small vs model error?)
  - GIF: prediction panel colors hot particles warm / frozen cool — you can
    watch time freeze in the quiet regions.
"""

import argparse
import time as _time

import numpy as np
import torch

from . import data as D
from .ckpt import eval_meta, load_checkpoint
from .rollout import render_gif
from .spines import make_spine


def _halo(pos, hot, radius):
    """hot mask -> hot ∪ (R-neighbors of hot). Chunked cdist, no autograd."""
    if hot.all() or not hot.any():
        return hot.clone()
    idx_hot = hot.nonzero(as_tuple=True)[0]
    mask = hot.clone()
    chunk = 2048
    for s in range(0, idx_hot.shape[0], chunk):
        d = torch.cdist(pos[idx_hot[s:s + chunk]], pos)
        mask |= (d < radius).any(0)
    return mask


@torch.no_grad()
def gated_rollout(model, traj, stats, meta, cfg, device, steps,
                  v_frac=0.35, coarse_every=8, dwell=4, gif_masks=None):
    """Returns (preds [T,N,D], stats dict). Mirrors rollout.rollout but with
    the two-rung integrator. Water-family ckpts only (no thermal/charge)."""
    wall_clip = cfg.get("wall_clip", 1.0)
    spine = make_spine(cfg)
    dim = cfg.get("dim", 2)
    n = traj.pos.shape[1]
    window = torch.tensor(traj.pos[: D.HISTORY + 1], dtype=torch.float32,
                          device=device)  # [C+1, N, D] rolling
    am = torch.tensor(stats["acc_mean"], dtype=torch.float32, device=device)
    astd = torch.tensor(stats["acc_std"], dtype=torch.float32, device=device)
    lo, hi = meta.domain
    margin = meta.wall_margin
    v_thresh = v_frac * float(np.mean(stats["vel_std"]))

    hot = torch.ones(n, dtype=torch.bool, device=device)  # start everyone hot
    dwell_left = torch.zeros(n, dtype=torch.long, device=device)
    preds = [window[i].cpu().numpy() for i in range(D.HISTORY + 1)]
    hot_frac_hist = []
    node_updates = 0

    for step in range(steps):
        p_now, p_prev = window[-1], window[-2]
        full_tick = (step % coarse_every == 0)
        if full_tick:
            sel = torch.ones(n, dtype=torch.bool, device=device)
        else:
            sel = _halo(p_now, hot, meta.radius)
        idx = sel.nonzero(as_tuple=True)[0]
        node_updates += int(idx.shape[0])

        p_next = p_now.clone()          # cold rung: frozen = its physics
        if idx.numel():                 # everyone frozen: nothing to compute
            sub_seq = window[:, idx]
            ei = D.build_edges(p_now[idx], meta.radius)
            nf = D.node_features(sub_seq, stats, meta, device,
                                 wall_clip=wall_clip)
            ef = D.edge_features(p_now[idx], ei, meta.radius)
            acc = model(nf, ei, ef) * astd + am
            if spine is not None:
                acc = acc + spine(p_now[idx])
            stepped = 2 * p_now[idx] - p_prev[idx] + acc
            if meta.walls:
                stepped = torch.clamp(stepped, lo + margin * 0.5,
                                      hi - margin * 0.5)
            p_next[idx] = stepped

        # reclassify from realized velocity; hysteresis via dwell
        v = (p_next - p_now).norm(dim=1)
        newly_hot = v > v_thresh
        dwell_left = torch.where(newly_hot,
                                 torch.full_like(dwell_left, dwell),
                                 (dwell_left - 1).clamp(min=0))
        hot = newly_hot | (dwell_left > 0)

        window = torch.cat([window[1:], p_next.unsqueeze(0)])
        preds.append(p_next.cpu().numpy())
        hot_frac_hist.append(float(hot.float().mean()))
        if gif_masks is not None:
            gif_masks.append(sel.cpu().numpy())

    info = {
        "hot_frac_mean": float(np.mean(hot_frac_hist)),
        "hot_frac_final": hot_frac_hist[-1],
        "node_updates": node_updates,
        "node_updates_full": steps * n,
    }
    return np.stack(preds), info


def main():
    from .rollout import rollout as full_rollout

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--traj", type=int, default=-1)
    ap.add_argument("--start", type=int, default=0,
                    help="seed the rollout this many frames in (settled-scene "
                         "regime is where property time earns its keep)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--v-frac", type=float, default=0.35,
                    help="hot threshold as a fraction of corpus vel_std")
    ap.add_argument("--coarse-every", type=int, default=8)
    ap.add_argument("--dwell", type=int, default=4)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg, stats, _ = load_checkpoint(args.ckpt, args.device)
    if cfg.get("thermal") or cfg.get("charged") or cfg.get("n_materials") \
            or cfg.get("n_regimes"):
        raise SystemExit("proptime supports single-material, non-thermal, "
                         "uncharged checkpoints only")
    meta = eval_meta(args.data, cfg)

    trajs = D.load_corpus(args.data)
    idx = args.traj if args.traj >= 0 else len(trajs) - 1
    truth = trajs[idx]
    if args.start:
        truth = D.Traj(pos=truth.pos[args.start:], mat=truth.mat,
                       temp=truth.temp[args.start:] if truth.temp is not None else None,
                       charge=truth.charge)
    steps = min(args.steps, truth.pos.shape[0] - D.HISTORY - 1)

    def timed(fn):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = _time.perf_counter()
        out = fn()
        if args.device == "cuda":
            torch.cuda.synchronize()
        return out, _time.perf_counter() - t0

    (full_pred, _), t_full = timed(
        lambda: full_rollout(model, truth, stats, meta, cfg, args.device, steps))
    masks = [] if args.gif else None
    (gated_pred, info), t_gated = timed(
        lambda: gated_rollout(model, truth, stats, meta, cfg, args.device,
                              steps, v_frac=args.v_frac,
                              coarse_every=args.coarse_every,
                              dwell=args.dwell, gif_masks=masks))

    m = min(len(full_pred), len(gated_pred), len(truth.pos))
    gate_err = float(np.mean((gated_pred[:m] - full_pred[:m]) ** 2))
    full_err = float(np.mean((full_pred[:m] - truth.pos[:m]) ** 2))
    gated_err = float(np.mean((gated_pred[:m] - truth.pos[:m]) ** 2))
    print(f"[proptime] traj {idx}, {steps} steps, device={args.device}")
    print(f"[proptime] wall-clock: full={t_full:.2f}s gated={t_gated:.2f}s "
          f"-> speedup {t_full / t_gated:.2f}x")
    print(f"[proptime] node updates: {info['node_updates']}/{info['node_updates_full']} "
          f"({100 * info['node_updates'] / info['node_updates_full']:.1f}%), "
          f"hot_frac mean={info['hot_frac_mean']:.3f} final={info['hot_frac_final']:.3f}")
    print(f"[proptime] MSE vs truth: full={full_err:.6f} gated={gated_err:.6f} | "
          f"gating deviation (gated vs full): {gate_err:.6f}")

    if args.gif:
        # color the prediction panel by rung: hot=warm, frozen=cool
        pad = [np.ones(gated_pred.shape[1], bool)] * (D.HISTORY + 1)
        heat = np.stack([m.astype(np.float32) for m in pad + masks])[:m]
        render_gif(truth.pos[:m], gated_pred[:m], args.gif,
                   truth_temps=None, pred_temps=heat, dim=cfg.get("dim", 2))
        print(f"[proptime] wrote {args.gif}")


if __name__ == "__main__":
    main()
