"""Training loop: normalized-acceleration MSE with noise injection.

--rollout-k 1: classic GNS one-step loss.
--rollout-k K>1: K-step unrolled loss — the model integrates its own
predictions and every step is scored against the clean trajectory (position
error / acc_std, which equals the one-step loss exactly at k=1). This attacks
accumulated drift directly (the Gen-2 float-up mechanism: at rest the residual
must cancel gravity EXACTLY; one-step training never feels the compounding).

Spines (exact known forces, see spines.py) are removed from the target; the
network learns only the residual. --spine auto picks nbody when the corpus
meta carries G (Gen 4), otherwise uniform gravity iff --gravity-residual.

Resumable: --resume loads latest.pt from --out if present, so a crashed or
interrupted run (GPU contention, reboot) continues from the last checkpoint.
"""

import argparse
import dataclasses
import json
import os
import time

# variable edge counts fragment the caching allocator toward full-card reserve;
# expandable segments keep the pool compact (must be set before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from . import data as D
from .model import GNS
from .spines import coulomb_spine, nbody_spine, uniform_spine


_OOM_ERROR = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def is_oom(err: RuntimeError) -> bool:
    """OOM in all its costumes. On ROCm, memory exhaustion often surfaces as
    hipBLAS handle/workspace allocation failures (HIPBLAS_STATUS_ALLOC_FAILED,
    or INTERNAL_ERROR under pressure) before torch's allocator can raise
    OutOfMemoryError — those must feed the degrade ladder, not crash the run
    (observed: the N-body corpus on a 16 GB RX 6800 XT crash-looped on
    batches a 24 GB card absorbed)."""
    # torch.OutOfMemoryError is top-level only from torch 2.5; older releases
    # expose the same class as torch.cuda.OutOfMemoryError.
    if isinstance(err, _OOM_ERROR):
        return True
    msg = str(err)
    return "HIPBLAS_STATUS_ALLOC_FAILED" in msg or (
        "HIPBLAS_STATUS_INTERNAL_ERROR" in msg
    ) or "out of memory" in msg.lower()


def resolve_spine(args, meta):
    """-> (spine_fn_or_factory, kind, g_frame, trap_w2_frame).

    Coulomb spines are per-TRAJECTORY (charge patterns vary), so the coulomb
    case returns a FACTORY taking a Traj; callers detect kind == 'coulomb'.
    """
    if args.spine == "none":
        return None, None, 0.0, 0.0
    if meta.charged and args.spine in ("auto", "coulomb"):
        k_frame = meta.K * meta.dt_record ** 2
        w2_frame = meta.trap_w2 * meta.dt_record ** 2
        def factory(traj):
            return coulomb_spine(k_frame, meta.soften, w2_frame,
                                 torch.tensor(traj.charge))
        return factory, "coulomb", k_frame, w2_frame
    if args.spine == "nbody" or (args.spine == "auto" and meta.G > 0):
        g_frame = meta.G * meta.dt_record ** 2
        return nbody_spine(g_frame, meta.soften), "nbody", g_frame, 0.0
    if args.spine == "uniform" or (args.spine == "auto" and args.gravity_residual):
        g_frame = D.gravity_per_frame(meta)
        return uniform_spine(g_frame, meta.dim), "uniform", g_frame, 0.0
    return None, None, 0.0, 0.0


def unrolled_loss(model, sample, stats, meta, args, spine, am, astd):
    """K-step pushforward loss for one sample; returns scalar loss tensor."""
    window, futures, temp, mat, reg, chg = sample
    W = window
    total = 0.0
    k_steps = futures.shape[0]
    for k in range(k_steps):
        p_now, p_prev = W[-1], W[-2]
        ei = D.build_edges(p_now.detach(), meta.radius)
        temp_win = temp[k: k + D.HISTORY + 1] if temp is not None else None
        nf = D.node_features(W, stats, meta, W.device,
                             wall_clip=args.wall_clip, temp_seq=temp_win,
                             charge=chg)
        ef = D.edge_features(p_now, ei, meta.radius)
        out = model(nf, ei, ef, mat=mat, regime=reg)
        acc = out[:, : meta.dim] * astd + am
        if spine is not None:
            acc = acc + spine(p_now)
        p_next = 2 * p_now - p_prev + acc
        total = total + ((p_next - futures[k]) / astd).pow(2).mean()
        if temp is not None and out.shape[1] > meta.dim:
            d_t_tgt = (temp[k + D.HISTORY + 1] - temp[k + D.HISTORY]
                       - stats["dtemp_mean"]) / stats["dtemp_std"]
            total = total + torch.nn.functional.mse_loss(out[:, meta.dim], d_t_tgt)
        W = torch.cat([W[1:], p_next.unsqueeze(0)])
    return total / k_steps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-final", type=float, default=1e-6)
    ap.add_argument("--noise", default="6.7e-4",
                    help="random-walk noise std, or 'auto' = 0.48 x mean vel_std "
                         "(the ratio the Gen-1 A/B validated for water)")
    ap.add_argument("--rollout-k", type=int, default=1,
                    help="unrolled-loss depth; 1 = classic one-step")
    ap.add_argument("--spine", choices=["auto", "none", "uniform", "nbody", "coulomb"],
                    default="auto")
    ap.add_argument("--gravity-residual", action="store_true",
                    help="exact-gravity spine: network learns residual accel only")
    ap.add_argument("--wall-clip", type=float, default=1.0,
                    help="wall visibility range in multiples of R (Gen 2+: 4.0)")
    ap.add_argument("--dynamic-frac", type=float, default=0.0,
                    help="fraction of samples drawn from early dynamic frames")
    ap.add_argument("--dynamic-cut", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--mat-dim", type=int, default=16)
    ap.add_argument("--conserve-momentum", action="store_true",
                    help="edge-force decoder: Newton's 3rd law by construction")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute block activations in backward: ~3-5x less "
                         "memory, ~30%% slower — dense frames fit small cards")
    ap.add_argument("--max-edges", type=int, default=0,
                    help="resample frames whose radius graph exceeds this edge "
                         "count (0 = off). Gravitating corpora go quadratic in "
                         "fully-merged end states — no VRAM fits them; better "
                         "to redraw cheaply and COUNT the censorship than to "
                         "burn OOM-churn on it")
    ap.add_argument("--holdout", default="2",
                    help="trajectories kept out of training; comma-separated "
                         "per corpus in merged mode (e.g. '10,2')")
    ap.add_argument("--stop-at", type=int, default=0,
                    help="stop (and checkpoint) at this step while keeping the "
                         "LR schedule defined by --steps (0 = run to --steps). "
                         "The LR decay horizon is --steps, so a run stopped "
                         "early is only reproduced with the same --steps")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # --data accepts comma-separated corpus dirs (Gen-10 data-level merge).
    # Each corpus keeps its own meta (radius!) and holdout; features are
    # unified to the merged width (thermal=any, n_materials=max) and every
    # sample is tagged with its corpus regime id.
    data_dirs = args.data.split(",")
    holdouts = [int(h) for h in str(args.holdout).split(",")]
    if len(holdouts) == 1:
        holdouts *= len(data_dirs)
    corpora = []
    for i, d in enumerate(data_dirs):
        m = D.load_meta(d)
        trs = D.load_corpus(d)
        train_trs = trs[: len(trs) - holdouts[i]] if holdouts[i] else trs
        corpora.append((m, train_trs))
        print(f"[train] corpus {i} ({d}): {len(train_trs)} train / {len(trs)} "
              f"total, radius={m.radius}, materials={m.n_materials}, "
              f"thermal={m.thermal}")
    merged = len(corpora) > 1
    base = corpora[0][0]
    if merged:
        for m, _ in corpora[1:]:
            assert (m.dim, m.walls, m.gravity, m.dt_record) == \
                   (base.dim, base.walls, base.gravity, base.dt_record), \
                   "merge requires agreeing dim/walls/gravity/dt_record"
    meta = dataclasses.replace(
        base,
        thermal=any(m.thermal for m, _ in corpora),
        n_materials=max(m.n_materials for m, _ in corpora),
    )
    # per-corpus meta with merged feature flags but native radius/margins
    feat_metas = [dataclasses.replace(m, thermal=meta.thermal,
                                      n_materials=meta.n_materials)
                  for m, _ in corpora]
    train_trajs = [tr for _, trs in corpora for tr in trs]  # pooled (stats)
    print(f"[train] merged={merged}, dim={meta.dim}, walls={meta.walls}, "
          f"materials={meta.n_materials}, thermal={meta.thermal}, "
          f"device={args.device}")

    spine, spine_kind, g_frame, trap_w2_frame = resolve_spine(args, meta)
    is_factory = spine_kind == "coulomb"
    if spine_kind:
        print(f"[train] spine={spine_kind} g_frame={g_frame:.3e} — "
              f"network learns the residual only")

    # Normalization stats. MERGED MODE IS PER-CORPUS. Pooling them caused a
    # 'wall-climb' defect in an earlier merged run: the multi-material and
    # thermal corpora's native scales differ 10x (vel_std_x) to 24x
    # (acc_std_x), so pooled constants crushed thermal targets in the loss and
    # amplified any shared output bias 24x at denorm time. The single-corpus
    # specialists were clean on the same held-out trajectories.
    stats_path = os.path.join(args.out, "stats.json")
    if os.path.exists(stats_path):
        stats = json.load(open(stats_path))
        if merged and not isinstance(stats, list):
            raise SystemExit(f"{stats_path} holds pooled (pre-fix) stats for a "
                             "merged run — use a fresh --out")
    else:
        if merged:
            stats = [D.compute_stats(trs,
                                     spine=None if is_factory else spine,
                                     spine_factory=spine if is_factory else None,
                                     thermal=m.thermal, device=args.device)
                     for m, trs in corpora]
            # temp-less corpora synthesize isothermal windows (make_sample);
            # they normalize with the thermal corpus's temp scale
            donor = next((s for s in stats if "temp_mean" in s), None)
            for s in stats:
                if donor is not None:
                    for k in ("temp_mean", "temp_std", "dtemp_mean", "dtemp_std"):
                        s.setdefault(k, donor[k])
        else:
            stats = D.compute_stats(train_trajs,
                                    spine=None if is_factory else spine,
                                    spine_factory=spine if is_factory else None,
                                    thermal=meta.thermal, device=args.device)
        json.dump(stats, open(stats_path, "w"), indent=2)
    stats_list = stats if merged else [stats]
    for i, s in enumerate(stats_list):
        print(f"[train] stats[{i}]: vel_std={s['vel_std']}, "
              f"acc_std={s['acc_std']}")

    if args.noise == "auto":
        noise_by_corpus = [0.48 * float(np.mean(s["vel_std"]))
                           for s in stats_list]
        for i, nz in enumerate(noise_by_corpus):
            print(f"[train] noise=auto -> corpus {i}: {nz:.3e}")
    else:
        noise_by_corpus = [float(args.noise)] * len(stats_list)

    node_in = D.node_feature_width(meta)
    out_dim = meta.dim + (1 if meta.thermal else 0)
    edge_in = meta.dim + 1
    n_regimes = len(corpora) if merged else 0
    model = GNS(node_in=node_in, edge_in=edge_in, hidden=args.hidden,
                layers=args.layers, out_dim=out_dim,
                n_materials=meta.n_materials, mat_dim=args.mat_dim,
                n_regimes=n_regimes,
                conserve_momentum=args.conserve_momentum,
                grad_checkpoint=args.grad_checkpoint).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_step = 0

    latest = os.path.join(args.out, "latest.pt")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"[train] resumed from step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model: {n_params/1e6:.2f}M params, node_in={node_in}, "
          f"out_dim={out_dim}, rollout_k={args.rollout_k}")

    ams = [torch.tensor(s["acc_mean"], dtype=torch.float32, device=args.device)
           for s in stats_list]
    astds = [torch.tensor(s["acc_std"], dtype=torch.float32, device=args.device)
             for s in stats_list]

    decay = (args.lr_final / args.lr) ** (1.0 / max(args.steps, 1))
    t0 = time.time()
    losses = []
    capped = 0
    frames = np.array([sum(len(tr) for tr in trs) for _, trs in corpora], float)
    corpus_w = frames / frames.sum()
    stop = min(args.steps, args.stop_at) if args.stop_at > 0 else args.steps
    for step in range(start_step, stop):
        for g in opt.param_groups:
            g["lr"] = args.lr * (decay ** step)

        samples = []
        for _ in range(args.batch):
            for _try in range(6):
                ci = int(rng.choice(len(corpora), p=corpus_w))
                c_meta, c_trajs = feat_metas[ci], corpora[ci][1]
                ti = int(rng.integers(0, len(c_trajs)))
                hi = c_trajs[ti].shape[0] - args.rollout_k
                if args.dynamic_frac and rng.random() < args.dynamic_frac:
                    t = int(rng.integers(D.HISTORY, min(args.dynamic_cut, hi)))
                else:
                    t = int(rng.integers(D.HISTORY, hi))
                if args.max_edges:
                    p = torch.tensor(c_trajs[ti].pos[t], dtype=torch.float32,
                                     device=args.device)
                    if D.build_edges(p, c_meta.radius).shape[1] > args.max_edges:
                        capped += 1
                        continue
                break
            spine_i = spine(c_trajs[ti]) if is_factory else spine
            c_stats = stats_list[ci if merged else 0]
            samples.append((
                D.make_sample(c_trajs[ti], t, c_stats, c_meta, args.device,
                              noise_by_corpus[ci if merged else 0],
                              wall_clip=args.wall_clip,
                              spine=spine_i, rollout_k=args.rollout_k,
                              regime=ci if merged else None),
                c_meta,
                spine_i,
                ci if merged else 0,
            ))
        # Impact-frame batches can spike edge counts (compressed water = dense
        # neighborhoods) and OOM the backward pass. Degrade gracefully:
        # full batch -> half batch -> skip, never crash the run.
        loss_val = None
        for attempt_samples in (samples, samples[:1]):
            try:
                if args.rollout_k > 1:
                    # per-sample backward: activation memory = one unroll
                    opt.zero_grad()
                    acc_loss = 0.0
                    for s, s_meta, s_spine, s_ci in attempt_samples:
                        loss = unrolled_loss(model, s, stats_list[s_ci],
                                             s_meta, args, s_spine,
                                             ams[s_ci], astds[s_ci])
                        (loss / len(attempt_samples)).backward()
                        acc_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    loss_val = acc_loss / len(attempt_samples)
                else:
                    nf, ei, ef, tgt, mat, reg = D.collate(
                        [s for s, _, _, _ in attempt_samples])
                    pred = model(nf, ei, ef, mat=mat, regime=reg)
                    loss = torch.nn.functional.mse_loss(pred, tgt)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    loss_val = loss.item()
                break
            except RuntimeError as err:
                if not is_oom(err):
                    raise
                opt.zero_grad(set_to_none=True)
                nf = ei = ef = tgt = mat = reg = pred = loss = None
                torch.cuda.empty_cache()
                print(f"[train] OOM at step {step+1} "
                      f"(batch {len(attempt_samples)}) — degrading", flush=True)
        if loss_val is None:
            continue
        losses.append(loss_val)

        if (step + 1) % args.log_every == 0:
            rate = args.log_every / (time.time() - t0)
            t0 = time.time()
            cap_note = f" edge-cap-resamples={capped}" if args.max_edges else ""
            print(f"[train] step {step+1}/{args.steps} loss={np.mean(losses):.5f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} {rate:.1f} it/s{cap_note}",
                  flush=True)
            losses = []

        if (step + 1) % args.ckpt_every == 0 or step + 1 == stop:
            ck = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step + 1,
                "stats": stats,
                "config": {
                    "node_in": node_in,
                    "edge_in": edge_in,
                    "out_dim": out_dim,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "radius": meta.radius,
                    "history": D.HISTORY,
                    "dim": meta.dim,
                    "walls": meta.walls,
                    "thermal": meta.thermal,
                    "n_materials": meta.n_materials,
                    "mat_dim": args.mat_dim,
                    "spine": spine_kind,
                    "g_frame": g_frame,
                    "soften": meta.soften,
                    "wall_clip": args.wall_clip,
                    "rollout_k": args.rollout_k,
                    "conserve_momentum": args.conserve_momentum,
                    "n_regimes": n_regimes,
                    "corpora": data_dirs,
                    "charged": meta.charged,
                    "trap_w2_frame": trap_w2_frame,
                },
            }
            torch.save(ck, latest + ".tmp")
            os.replace(latest + ".tmp", latest)

    print("[train] done")


if __name__ == "__main__":
    main()
