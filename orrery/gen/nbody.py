"""2D N-body planetesimal teacher (pure torch, no Taichi) — Gen 4: The Heavens.

Self-gravitating cloud of equal-mass particles: softened pairwise gravity +
sticky soft-sphere contacts (spring + near-critical normal damping, so
collisions dissipate and bodies ACCRETE — planetesimals clump into planets
with fixed N; a "planet" is a bound clump).

The student's exact spine is the gravity term (spines.nbody_spine — same
formula, same softening); the network learns only the contact/collision
residual, which is short-range and lives on the radius graph.

Scenes: rotating disk, sub-Keplerian factor randomized per trajectory —
low spin collapses hard, high spin flattens into rings before clumping.

Runs anywhere torch runs (no Taichi dependency).
Output: traj_*.npz (positions [T,N,2]) + meta.json with G/soften recorded so
training and rollout build the identical spine.
"""

import argparse
import json
import os

import numpy as np
import torch


def pair_forces(pos, vel, G, soften, r_p, k_n, c_n, chunk=4096):
    """Gravity + sticky contact accelerations, chunked over sources."""
    n = pos.shape[0]
    acc = torch.zeros_like(pos)
    eps2 = soften * soften
    contact_d = 2.0 * r_p
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rel = pos.unsqueeze(0) - pos[s:e].unsqueeze(1)      # [c, N, 2] toward others
        d2 = (rel * rel).sum(-1) + eps2
        inv = d2.pow(-1.5)
        inv[:, s:e].fill_diagonal_(0.0)
        acc[s:e] += G * (rel * inv.unsqueeze(-1)).sum(1)     # gravity (attractive)

        d = d2.sqrt()                                        # ~distance (softened)
        overlap = contact_d - d
        mask = overlap > 0
        mask[:, s:e].fill_diagonal_(False)
        if mask.any():
            nrm = rel / d.unsqueeze(-1)                      # unit, toward other
            # spring pushes APART (negative along nrm)
            f_spring = -k_n * overlap.clamp(min=0.0)
            # damping on the normal relative velocity (sticky: kills bounce)
            vrel = vel.unsqueeze(0) - vel[s:e].unsqueeze(1)  # other - self
            v_n = (vrel * nrm).sum(-1)
            f_damp = c_n * v_n                               # approach: v_n < 0
            f = torch.where(mask, f_spring + f_damp, torch.zeros_like(d))
            acc[s:e] += (nrm * f.unsqueeze(-1)).sum(1)
    return acc


def init_disk(rng, n, radius=0.35, spin_frac=0.6, v_noise=0.05, G=1.3e-4):
    """Rotating disk, sub-Keplerian → collapse with angular momentum."""
    r = radius * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    pos = np.stack([0.5 + r * np.cos(th), 0.5 + r * np.sin(th)], 1)
    a_char = G * n / max(radius, 1e-6) ** 2
    omega = spin_frac * np.sqrt(a_char / radius)
    vel = omega * np.stack([-(pos[:, 1] - 0.5), pos[:, 0] - 0.5], 1)
    vel += v_noise * omega * radius * rng.standard_normal((n, 2))
    vel -= vel.mean(0)  # zero net momentum: cloud stays centered
    return (pos.astype(np.float32), vel.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj", type=int, default=30)
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--substeps-per-frame", type=int, default=40)
    ap.add_argument("--dt-record", type=float, default=3e-3)
    ap.add_argument("--G", type=float, default=1.3e-4)
    ap.add_argument("--soften", type=float, default=0.004)
    ap.add_argument("--r-p", type=float, default=0.004)
    ap.add_argument("--k-n", type=float, default=2e4)
    ap.add_argument("--c-n", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dt = args.dt_record / args.substeps_per_frame
    dev = args.device
    print(f"[gen] nbody: N={args.particles}, G={args.G}, dt_sub={dt:.2e}, "
          f"device={dev}")

    for t in range(args.traj):
        rng = np.random.default_rng(args.seed * 10_000 + t)
        spin = float(rng.uniform(0.35, 0.85))
        pos_np, vel_np = init_disk(rng, args.particles, spin_frac=spin, G=args.G)
        pos = torch.tensor(pos_np, device=dev)
        vel = torch.tensor(vel_np, device=dev)

        frames = np.empty((args.frames, args.particles, 2), dtype=np.float32)
        acc = pair_forces(pos, vel, args.G, args.soften, args.r_p,
                          args.k_n, args.c_n)
        bad = False
        for f in range(args.frames):
            for _ in range(args.substeps_per_frame):
                # velocity Verlet (kick-drift-kick)
                vel = vel + 0.5 * dt * acc
                pos = pos + dt * vel
                acc = pair_forces(pos, vel, args.G, args.soften, args.r_p,
                                  args.k_n, args.c_n)
                vel = vel + 0.5 * dt * acc
            frames[f] = pos.cpu().numpy()
            if not np.isfinite(frames[f]).all():
                bad = True
                break
        if bad:
            print(f"[gen] traj {t}: NaN/inf detected, skipping")
            continue
        np.savez_compressed(os.path.join(args.out, f"traj_{t:04d}.npz"),
                            positions=frames)
        print(f"[gen] traj {t}: {args.frames} frames, spin={spin:.2f} "
              f"[{t+1}/{args.traj}]", flush=True)

    meta = {
        "n_particles": args.particles,
        "frames": args.frames,
        "dt_record": args.dt_record,
        "mean_spacing": args.r_p * 2,
        # student graph radius: see approaching neighbors before contact, but
        # sized for POST-collapse density — gravity crushes clump spacing below
        # the contact diameter, and 5*r_p exploded edge counts into 24GB OOM
        # territory on a 24 GB card. 3.5*r_p = 1.75 contact diameters.
        "radius": round(3.5 * args.r_p, 6),
        "domain": [0.0, 1.0],
        "wall_margin": 0.0,
        "dim": 2,
        "walls": False,
        "gravity": 0.0,
        "G": args.G,
        "soften": args.soften,
        "r_p": args.r_p,
        "n_materials": 0,
        "thermal": False,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[gen] wrote meta.json: radius={meta['radius']} (spine: nbody)")


if __name__ == "__main__":
    main()
