"""3D MLS-MPM water trajectory generator (Taichi) — Gen 6 teacher.

mpm88-style weakly-compressible water lifted to 3D on a unit cube. Each
trajectory drops a randomly sized/placed water box with a random initial
velocity. Same recording conventions as the 2D family (dt_record, meta),
dim=3 — the shared data/model/train stack handles the rest.

Radius convention: 2.0 x mean spacing (not 2.5 as in 2D) — 3D neighborhood
volume grows cubically and the edge budget is the training cost driver.
"""

# Derived from the Taichi example python/taichi/examples/simulation/mpm88.py
# (https://github.com/taichi-dev/taichi), Copyright (c) 2016 The Taichi
# Authors, licensed under the Apache License, Version 2.0. Modified for
# Orrery. See THIRD_PARTY_NOTICES for the license text.
# The MLS-MPM water solver is derived from that 2D example and lifted to 3D;
# the trajectory generator around it (random scenes, recording, meta.json,
# CLI) is new.

import argparse
import json
import os

import numpy as np


def build_sim(ti, n_particles: int, n_grid: int = 64):
    dx = 1.0 / n_grid
    inv_dx = float(n_grid)
    dt = 1e-4
    p_rho = 1.0
    p_vol = (dx * 0.5) ** 3
    p_mass = p_vol * p_rho
    E = 400.0

    x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
    J = ti.field(dtype=ti.f32, shape=n_particles)
    grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

    @ti.kernel
    def substep():
        for i, j, k in grid_m:
            grid_v[i, j, k] = [0.0, 0.0, 0.0]
            grid_m[i, j, k] = 0.0
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            stress = -dt * 4.0 * E * p_vol * (J[p] - 1.0) * inv_dx * inv_dx
            affine = ti.Matrix.identity(ti.f32, 3) * stress + p_mass * C[p]
            for ii, jj, kk in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([ii, jj, kk])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[ii][0] * w[jj][1] * w[kk][2]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
        for i, j, k in grid_m:
            if grid_m[i, j, k] > 0:
                grid_v[i, j, k] /= grid_m[i, j, k]
            grid_v[i, j, k][1] -= dt * 9.8  # y is up
            if i < 3 and grid_v[i, j, k][0] < 0:
                grid_v[i, j, k][0] = 0
            if i > n_grid - 3 and grid_v[i, j, k][0] > 0:
                grid_v[i, j, k][0] = 0
            if j < 3 and grid_v[i, j, k][1] < 0:
                grid_v[i, j, k][1] = 0
            if j > n_grid - 3 and grid_v[i, j, k][1] > 0:
                grid_v[i, j, k][1] = 0
            if k < 3 and grid_v[i, j, k][2] < 0:
                grid_v[i, j, k][2] = 0
            if k > n_grid - 3 and grid_v[i, j, k][2] > 0:
                grid_v[i, j, k][2] = 0
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(ti.f32, 3)
            new_C = ti.Matrix.zero(ti.f32, 3, 3)
            for ii, jj, kk in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([ii, jj, kk])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[ii][0] * w[jj][1] * w[kk][2]
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4.0 * weight * g_v.outer_product(dpos) * inv_dx
            v[p] = new_v
            C[p] = new_C
            x[p] += dt * v[p]
            J[p] *= 1.0 + dt * (new_C[0, 0] + new_C[1, 1] + new_C[2, 2])

    def init_block(rng: np.random.Generator):
        bw = rng.uniform(0.22, 0.40)
        bh = rng.uniform(0.22, 0.40)
        bd = rng.uniform(0.22, 0.40)
        bx = rng.uniform(0.08, 0.92 - bw)
        by = rng.uniform(0.30, 0.92 - bh)
        bz = rng.uniform(0.08, 0.92 - bd)
        pos = np.empty((n_particles, 3), dtype=np.float32)
        pos[:, 0] = rng.uniform(bx, bx + bw, n_particles)
        pos[:, 1] = rng.uniform(by, by + bh, n_particles)
        pos[:, 2] = rng.uniform(bz, bz + bd, n_particles)
        vel = np.tile(
            rng.uniform([-1.0, -1.0, -1.0], [1.0, 0.5, 1.0]).astype(np.float32),
            (n_particles, 1),
        )
        x.from_numpy(pos)
        v.from_numpy(vel)
        C.from_numpy(np.zeros((n_particles, 3, 3), dtype=np.float32))
        J.from_numpy(np.ones(n_particles, dtype=np.float32))
        return float((bw * bh * bd / n_particles) ** (1.0 / 3.0))

    return substep, init_block, x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj", type=int, default=40)
    ap.add_argument("--particles", type=int, default=4000)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--substeps-per-frame", type=int, default=30)
    ap.add_argument("--n-grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=["auto", "vulkan", "cpu"], default="auto")
    args = ap.parse_args()

    import taichi as ti

    arch = None
    if args.arch in ("auto", "vulkan"):
        try:
            ti.init(arch=ti.vulkan, offline_cache=True)
            arch = "vulkan"
        except Exception as e:
            if args.arch == "vulkan":
                raise
            print(f"[gen] vulkan unavailable ({e}); falling back to cpu")
    if arch is None:
        ti.init(arch=ti.cpu, offline_cache=True)
        arch = "cpu"
    print(f"[gen] taichi arch={arch} (3D, grid {args.n_grid}^3)")

    os.makedirs(args.out, exist_ok=True)
    substep, init_block, x = build_sim(ti, args.particles, n_grid=args.n_grid)

    spacings = []
    for t in range(args.traj):
        rng = np.random.default_rng(args.seed * 10_000 + t)
        spacing = init_block(rng)
        spacings.append(spacing)
        frames = np.empty((args.frames, args.particles, 3), dtype=np.float32)
        for f in range(args.frames):
            for _ in range(args.substeps_per_frame):
                substep()
            frames[f] = x.to_numpy()
        if not np.isfinite(frames).all():
            print(f"[gen] traj {t}: NaN/inf detected, skipping")
            continue
        np.savez_compressed(os.path.join(args.out, f"traj_{t:04d}.npz"),
                            positions=frames)
        print(f"[gen] traj {t}: saved {args.frames} frames, "
              f"spacing={spacing:.5f}", flush=True)

    mean_spacing = float(np.mean(spacings))
    meta = {
        "n_particles": args.particles,
        "frames": args.frames,
        "dt_record": 1e-4 * args.substeps_per_frame,
        "mean_spacing": mean_spacing,
        "radius": round(2.0 * mean_spacing, 6),
        "domain": [0.0, 1.0],
        "wall_margin": 3.0 / args.n_grid,
        "arch": arch,
        "dim": 3,
        "walls": True,
        "gravity": 9.8,
        "n_materials": 0,
        "thermal": False,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[gen] wrote meta.json: radius={meta['radius']}")


if __name__ == "__main__":
    main()
