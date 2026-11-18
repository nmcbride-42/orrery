"""2D charged-particle plasma teacher (pure torch) — Gen 9 first light.

Neutral two-species cloud: softened Coulomb (like repels, unlike attracts),
sticky soft-sphere contacts, and a weak harmonic trap toward the domain
center (a Paul-trap stand-in — without it, like-charge pressure disperses
the neutral cloud out of the domain). Coulomb + trap are EXACT laws and form
the student's spine (per-trajectory: the charge pattern varies); the network
learns only the contact residual, with charge as a node feature.

Runs anywhere torch runs (no Taichi dependency).
Output: traj_*.npz (positions [T,N,2], charges [N]) + meta.json carrying
K/soften/trap_w2 so training and rollout rebuild the identical spine.
"""

import argparse
import json
import os

import numpy as np
import torch


def pair_forces(pos, vel, q, K, soften, r_p, k_n, c_n, trap_w2, chunk=4096):
    """Signed Coulomb + sticky contact + harmonic trap accelerations."""
    n = pos.shape[0]
    acc = -trap_w2 * (pos - 0.5)
    eps2 = soften * soften
    contact_d = 2.0 * r_p
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rel = pos[s:e].unsqueeze(1) - pos.unsqueeze(0)      # [c, N, 2] away from others
        d2 = (rel * rel).sum(-1) + eps2
        inv = d2.pow(-1.5)
        inv[:, s:e].fill_diagonal_(0.0)
        qq = q[s:e].unsqueeze(1) * q.unsqueeze(0)            # [c, N] sign of pair
        # like charges (+) push along rel (= away): repulsion; unlike attract
        acc[s:e] += K * (qq.unsqueeze(-1) * rel * inv.unsqueeze(-1)).sum(1)

        d = d2.sqrt()
        overlap = contact_d - d
        mask = overlap > 0
        mask[:, s:e].fill_diagonal_(False)
        if mask.any():
            nrm = rel / d.unsqueeze(-1)                      # unit, away from other
            f_spring = k_n * overlap.clamp(min=0.0)          # pushes apart
            vrel = vel[s:e].unsqueeze(1) - vel.unsqueeze(0)
            v_n = (vrel * nrm).sum(-1)
            f_damp = -c_n * v_n
            f = torch.where(mask, f_spring + f_damp, torch.zeros_like(d))
            acc[s:e] += (nrm * f.unsqueeze(-1)).sum(1)
    return acc


def init_cloud(rng, n, radius=0.3, spin=0.4, v_noise=0.08, char_a=2.0):
    r = radius * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    pos = np.stack([0.5 + r * np.cos(th), 0.5 + r * np.sin(th)], 1)
    omega = spin * np.sqrt(char_a / radius)
    vel = omega * np.stack([-(pos[:, 1] - 0.5), pos[:, 0] - 0.5], 1)
    vel += v_noise * omega * radius * rng.standard_normal((n, 2))
    vel -= vel.mean(0)
    q = np.ones(n, dtype=np.float32)
    q[rng.permutation(n)[: n // 2]] = -1.0  # exactly neutral
    return pos.astype(np.float32), vel.astype(np.float32), q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj", type=int, default=30)
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--substeps-per-frame", type=int, default=40)
    ap.add_argument("--dt-record", type=float, default=3e-3)
    ap.add_argument("--K", type=float, default=1.3e-4)
    ap.add_argument("--soften", type=float, default=0.004)
    ap.add_argument("--r-p", type=float, default=0.004)
    ap.add_argument("--k-n", type=float, default=2e4)
    ap.add_argument("--c-n", type=float, default=150.0)
    ap.add_argument("--trap-w2", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dt = args.dt_record / args.substeps_per_frame
    dev = args.device
    print(f"[gen] charged: N={args.particles}, K={args.K}, trap_w2={args.trap_w2}, "
          f"dt_sub={dt:.2e}, device={dev}")

    for t in range(args.traj):
        rng = np.random.default_rng(args.seed * 10_000 + t)
        spin = float(rng.uniform(0.2, 0.7))
        char_a = args.K * args.particles / 0.09
        pos_np, vel_np, q_np = init_cloud(rng, args.particles, spin=spin,
                                          char_a=max(char_a, 1e-6))
        pos = torch.tensor(pos_np, device=dev)
        vel = torch.tensor(vel_np, device=dev)
        q = torch.tensor(q_np, device=dev)

        frames = np.empty((args.frames, args.particles, 2), dtype=np.float32)
        acc = pair_forces(pos, vel, q, args.K, args.soften, args.r_p,
                          args.k_n, args.c_n, args.trap_w2)
        bad = False
        for f in range(args.frames):
            for _ in range(args.substeps_per_frame):
                vel = vel + 0.5 * dt * acc
                pos = pos + dt * vel
                acc = pair_forces(pos, vel, q, args.K, args.soften, args.r_p,
                                  args.k_n, args.c_n, args.trap_w2)
                vel = vel + 0.5 * dt * acc
            frames[f] = pos.cpu().numpy()
            if not np.isfinite(frames[f]).all():
                bad = True
                break
        if bad:
            print(f"[gen] traj {t}: NaN/inf detected, skipping")
            continue
        np.savez_compressed(os.path.join(args.out, f"traj_{t:04d}.npz"),
                            positions=frames, charges=q_np)
        print(f"[gen] traj {t}: {args.frames} frames, spin={spin:.2f} "
              f"[{t+1}/{args.traj}]", flush=True)

    meta = {
        "n_particles": args.particles,
        "frames": args.frames,
        "dt_record": args.dt_record,
        "mean_spacing": args.r_p * 2,
        # post-collapse density lesson from gen4 applies here too
        "radius": round(3.5 * args.r_p, 6),
        "domain": [0.0, 1.0],
        "wall_margin": 0.0,
        "dim": 2,
        "walls": False,
        "gravity": 0.0,
        "charged": True,
        "K": args.K,
        "soften": args.soften,
        "trap_w2": args.trap_w2,
        "r_p": args.r_p,
        "n_materials": 0,
        "thermal": False,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[gen] wrote meta.json: radius={meta['radius']} (spine: coulomb)")


if __name__ == "__main__":
    main()
