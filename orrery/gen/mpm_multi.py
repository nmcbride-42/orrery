"""2D multi-material MLS-MPM trajectory generator (Taichi) — Gen 3 teacher.

mpm99-family solver: F-based with SVD, three materials on one grid —
  0 water (mu=0, F reset to sqrt(J)·I)
  1 jelly (neo-Hookean, softened h=0.3)
  2 snow  (plasticity via clamped singular values + hardening)

Scenes: 2-3 non-overlapping random blocks, each a random material, with at
least two distinct materials per scene (cross-material contact is the whole
point — it is the only place inter-material physics is learnable from).
`--pure-water` emits all-water scenes for the holdout eval (float-up verdict).

`--thermal` (Gen 5 teacher): per-particle temperature advected with the
material point, grid-based diffusion, Boussinesq buoyancy, hot floor / cold
ceiling. Scene becomes a resting pool (with warmup) so convection cells are
the signal, not splash dynamics.

Output: one .npz per trajectory (positions [T,N,2], materials [N] uint8,
temps [T,N] if thermal) plus meta.json (radius, dt, materials, flags).
"""

# Derived from the Taichi example python/taichi/examples/simulation/mpm99.py
# (https://github.com/taichi-dev/taichi), Copyright (c) 2016 The Taichi
# Authors, licensed under the Apache License, Version 2.0. Modified for
# Orrery. See THIRD_PARTY_NOTICES for the license text.
# The multi-material solver (SVD-based F update, snow plasticity clamp and
# hardening, jelly/water handling) is derived from that example; the thermal
# transport, scene generation, recording, meta.json and CLI are new.

import argparse
import json
import os

import numpy as np

WATER, JELLY, SNOW = 0, 1, 2
MATERIAL_NAMES = ["water", "jelly", "snow"]


def build_sim(ti, n_particles: int, n_grid: int = 128, thermal: bool = False,
              kappa: float = 0.04, buoyancy: float = 40.0, t_ref: float = 0.5,
              surface_cool: float = 30.0, e_mod: float = 1000.0):
    dx = 1.0 / n_grid
    inv_dx = float(n_grid)
    dt = 1e-4
    p_rho = 1.0
    p_vol = (dx * 0.5) ** 2
    p_mass = p_vol * p_rho
    E, nu = e_mod, 0.2
    mu_0 = E / (2.0 * (1.0 + nu))
    la_0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
    v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
    C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)
    F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)
    Jp = ti.field(dtype=ti.f32, shape=n_particles)
    mat = ti.field(dtype=ti.i32, shape=n_particles)
    T = ti.field(dtype=ti.f32, shape=n_particles)
    grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid, n_grid))
    grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
    grid_t = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
    grid_t2 = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
    grid_t0 = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
    # floor heat patches [lo0, hi0, lo1, hi1] in grid-i units; hi < lo disables.
    # Patch heating (vs whole-floor) breaks the lateral symmetry so circulation
    # is FORCED by baroclinic torque, not waiting on an instability to outgrow
    # MPM's numerical dissipation (smoke finding: uniform floor -> static
    # stratified equilibrium, motion dies).
    patch = ti.field(dtype=ti.i32, shape=4)

    @ti.kernel
    def substep():
        for i, j in grid_m:
            grid_v[i, j] = [0.0, 0.0]
            grid_m[i, j] = 0.0
            grid_t[i, j] = 0.0
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]
            h = ti.exp(10.0 * (1.0 - Jp[p]))  # snow hardens when compressed
            if mat[p] == JELLY:
                h = 0.3
            mu, la = mu_0 * h, la_0 * h
            if mat[p] == WATER:
                mu = 0.0
            U, sig, V = ti.svd(F[p])
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                if mat[p] == SNOW:
                    new_sig = ti.min(ti.max(sig[d, d], 1.0 - 2.5e-2), 1.0 + 4.5e-3)
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if mat[p] == WATER:
                F[p] = ti.Matrix.identity(ti.f32, 2) * ti.sqrt(J)
            elif mat[p] == SNOW:
                F[p] = U @ sig @ V.transpose()
            stress = (
                2.0 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose()
                + ti.Matrix.identity(ti.f32, 2) * la * J * (J - 1.0)
            )
            stress = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * stress
            affine = stress + p_mass * C[p]
            for ii, jj in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([ii, jj])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[ii][0] * w[jj][1]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
                if ti.static(thermal):
                    grid_t[base + offset] += weight * p_mass * T[p]
        for i, j in grid_m:
            if grid_m[i, j] > 0:
                grid_v[i, j] /= grid_m[i, j]
                if ti.static(thermal):
                    grid_t[i, j] /= grid_m[i, j]
            grid_v[i, j][1] -= dt * 9.8
            if ti.static(thermal):
                # Boussinesq buoyancy against the reference temperature
                # (pre-BC temps: contact heating reaches buoyancy next substep)
                if grid_m[i, j] > 0:
                    grid_v[i, j][1] += dt * buoyancy * (grid_t[i, j] - t_ref)
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
        if ti.static(thermal):
            # FLIP-style delta transport: snapshot the scattered fluid T, then
            # apply BC + diffusion + cooling; particles receive only the CHANGE.
            # PIC-style full replacement is a ~1.5-cell box blur per substep —
            # enough numerical diffusion to flood the pool from the floor in
            # ~25 frames and kill the gradient convection needs (smoke finding).
            for i, j in grid_t:
                grid_t0[i, j] = grid_t[i, j]
                # hot floor PATCHES / cold ceiling boundary condition
                if j < 4 and ((patch[0] <= i <= patch[1]) or (patch[2] <= i <= patch[3])):
                    grid_t[i, j] = 1.0
                if j > n_grid - 5:
                    grid_t[i, j] = 0.0
            # explicit diffusion on the grid (kappa*dt/dx^2 must stay < 0.25).
            # Empty cells are NO-FLUX, not T=0 — otherwise the vacuum above the
            # pool drains all heat (smoke-test finding). Floor BC rows count as
            # valid sources so the hot floor conducts into the fluid.
            for i, j in grid_t:
                lap = 0.0
                if 0 < i < n_grid - 1 and 0 < j < n_grid - 1 and grid_m[i, j] > 0:
                    tc = grid_t[i, j]
                    tl = grid_t[i - 1, j] if grid_m[i - 1, j] > 0 else tc
                    tr = grid_t[i + 1, j] if grid_m[i + 1, j] > 0 else tc
                    tb = grid_t[i, j - 1] if (grid_m[i, j - 1] > 0 or j - 1 < 4) else tc
                    tt = grid_t[i, j + 1] if grid_m[i, j + 1] > 0 else tc
                    lap = tl + tr + tb + tt - 4.0 * tc
                grid_t2[i, j] = grid_t[i, j] + dt * kappa * inv_dx * inv_dx * lap
                # free-surface radiative cooling: without a cold sink the pool
                # saturates hot and convection stalls (smoke-test finding)
                if grid_m[i, j] > 0 and j < n_grid - 1 and grid_m[i, j + 1] == 0:
                    grid_t2[i, j] -= dt * surface_cool * grid_t2[i, j]
            for i, j in grid_t:
                grid_t[i, j] = grid_t2[i, j]
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(ti.f32, 2)
            new_C = ti.Matrix.zero(ti.f32, 2, 2)
            new_t = 0.0
            t_wsum = 0.0
            for ii, jj in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([ii, jj])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[ii][0] * w[jj][1]
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4.0 * weight * g_v.outer_product(dpos) * inv_dx
                if ti.static(thermal):
                    # gather the T DELTA only from cells that hold fluid (or
                    # the hot floor BC rows) — empty cells carry no signal
                    gi = base + offset
                    if grid_m[gi] > 0 or gi[1] < 4:
                        new_t += weight * (grid_t[gi] - grid_t0[gi])
                        t_wsum += weight
            v[p] = new_v
            C[p] = new_C
            x[p] += dt * v[p]
            if ti.static(thermal):
                if t_wsum > 1e-6:
                    T[p] = ti.min(ti.max(T[p] + new_t / t_wsum, 0.0), 1.0)

    def reset(pos: np.ndarray, vel: np.ndarray, mats: np.ndarray, temps: np.ndarray,
              patches=(0, -1, 0, -1)):
        x.from_numpy(pos)
        v.from_numpy(vel)
        mat.from_numpy(mats.astype(np.int32))
        T.from_numpy(temps)
        C.from_numpy(np.zeros((n_particles, 2, 2), dtype=np.float32))
        F.from_numpy(np.tile(np.eye(2, dtype=np.float32), (n_particles, 1, 1)))
        Jp.from_numpy(np.ones(n_particles, dtype=np.float32))
        patch.from_numpy(np.array(patches, dtype=np.int32))

    return substep, reset, x, T


def place_blocks(rng, n_blocks: int):
    """Non-overlapping random boxes (x, y, w, h), rejection-sampled."""
    boxes = []
    for _ in range(n_blocks):
        for _attempt in range(200):
            bw = rng.uniform(0.14, 0.30)
            bh = rng.uniform(0.14, 0.30)
            bx = rng.uniform(0.08, 0.92 - bw)
            by = rng.uniform(0.20, 0.92 - bh)
            pad = 0.02
            if all(
                bx + bw + pad < ox or ox + ow + pad < bx
                or by + bh + pad < oy or oy + oh + pad < by
                for ox, oy, ow, oh in boxes
            ):
                boxes.append((bx, by, bw, bh))
                break
        else:
            break  # couldn't place more; scene proceeds with fewer blocks
    return boxes


def init_scene(rng, n_particles: int, pure_water: bool, thermal: bool,
               n_grid: int = 128):
    """Returns (pos, vel, mats, temps, mean_spacing)."""
    if thermal:
        # resting pool: convection is the signal, not splash. Pool dimensions
        # are density-computed so particle spacing ~= 0.75*dx — sparser pools
        # have too little grid mass for pressure to resist gravity and crush
        # to the floor like dust (smoke-test finding).
        spacing_target = 0.75 / n_grid
        area = n_particles * spacing_target ** 2
        width = min(0.9, area / 0.15)
        height = area / width
        x_lo, x_hi = 0.5 - width / 2, 0.5 + width / 2
        y_lo, y_hi = 0.05, 0.05 + height
        pos = np.empty((n_particles, 2), dtype=np.float32)
        pos[:, 0] = rng.uniform(x_lo, x_hi, n_particles)
        pos[:, 1] = rng.uniform(y_lo, y_hi, n_particles)
        vel = np.zeros((n_particles, 2), dtype=np.float32)
        mats = np.zeros(n_particles, dtype=np.uint8)
        # cold-ish start with noise so the instability has a seed
        temps = (0.35 + 0.1 * rng.random(n_particles)).astype(np.float32)
        area = (x_hi - x_lo) * (y_hi - y_lo)
        # 1-2 random floor heat patches, in grid-i units, inside the pool span
        n_patch = int(rng.integers(1, 3))
        i_lo, i_hi = int(x_lo * n_grid), int(x_hi * n_grid)
        patches = [0, -1, 0, -1]
        for k in range(n_patch):
            w = int(rng.uniform(0.15, 0.30) * n_grid)
            lo = int(rng.integers(i_lo, max(i_lo + 1, i_hi - w)))
            patches[2 * k], patches[2 * k + 1] = lo, lo + w
        return pos, vel, mats, temps, float(np.sqrt(area / n_particles)), tuple(patches)

    n_blocks = int(rng.integers(2, 4))
    boxes = place_blocks(rng, n_blocks)
    if pure_water:
        block_mats = [WATER] * len(boxes)
    else:
        while True:
            block_mats = [int(rng.integers(0, 3)) for _ in boxes]
            if len(set(block_mats)) >= min(2, len(boxes)):
                break
    areas = np.array([w * h for _, _, w, h in boxes])
    counts = np.maximum((areas / areas.sum() * n_particles).astype(int), 1)
    counts[-1] = n_particles - counts[:-1].sum()  # exact total

    pos = np.empty((n_particles, 2), dtype=np.float32)
    vel = np.empty((n_particles, 2), dtype=np.float32)
    mats = np.empty(n_particles, dtype=np.uint8)
    at = 0
    for (bx, by, bw, bh), m, c in zip(boxes, block_mats, counts):
        pos[at:at + c, 0] = rng.uniform(bx, bx + bw, c)
        pos[at:at + c, 1] = rng.uniform(by, by + bh, c)
        vel[at:at + c] = rng.uniform([-1.0, -1.0], [1.0, 0.5]).astype(np.float32)
        mats[at:at + c] = m
        at += c
    temps = np.zeros(n_particles, dtype=np.float32)
    total_area = float(areas.sum())
    return (pos, vel, mats, temps, float(np.sqrt(total_area / n_particles)),
            (0, -1, 0, -1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj", type=int, default=60)
    ap.add_argument("--pure-water-traj", type=int, default=8,
                    help="all-water holdout scenes appended AFTER the mixed ones "
                         "(sorted last, so --holdout N keeps them out of training)")
    ap.add_argument("--particles", type=int, default=3000)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--substeps-per-frame", type=int, default=30)
    ap.add_argument("--warmup-frames", type=int, default=0,
                    help="unrecorded settle frames before capture (thermal pool)")
    ap.add_argument("--thermal", action="store_true")
    ap.add_argument("--buoyancy", type=float, default=40.0)
    ap.add_argument("--kappa", type=float, default=0.04)
    ap.add_argument("--surface-cool", type=float, default=30.0)
    ap.add_argument("--n-grid", type=int, default=0,
                    help="0 = auto (128, or 64 for thermal pools)")
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
    print(f"[gen] taichi arch={arch} thermal={args.thermal}")

    os.makedirs(args.out, exist_ok=True)
    n_grid = args.n_grid or (64 if args.thermal else 128)
    substep, reset, x, T = build_sim(
        ti, args.particles, n_grid=n_grid, thermal=args.thermal,
        kappa=args.kappa, buoyancy=args.buoyancy, surface_cool=args.surface_cool,
        e_mod=400.0 if args.thermal else 1000.0,  # softer columns circulate
    )
    if args.thermal:
        args.warmup_frames = args.warmup_frames or 50

    plan = [(t, False) for t in range(args.traj)]
    if not args.thermal:
        plan += [(args.traj + i, True) for i in range(args.pure_water_traj)]

    spacings = []
    for idx, (t, pure) in enumerate(plan):
        rng = np.random.default_rng(args.seed * 10_000 + t)
        pos0, vel0, mats, temps0, spacing, patches = init_scene(
            rng, args.particles, pure_water=pure, thermal=args.thermal,
            n_grid=n_grid,
        )
        reset(pos0, vel0, mats, temps0, patches=patches)
        spacings.append(spacing)
        for _ in range(args.warmup_frames * args.substeps_per_frame):
            substep()
        frames = np.empty((args.frames, args.particles, 2), dtype=np.float32)
        temps = (np.empty((args.frames, args.particles), dtype=np.float32)
                 if args.thermal else None)
        for f in range(args.frames):
            for _ in range(args.substeps_per_frame):
                substep()
            frames[f] = x.to_numpy()
            if args.thermal:
                temps[f] = T.to_numpy()
        if not np.isfinite(frames).all():
            print(f"[gen] traj {t}: NaN/inf detected, skipping")
            continue
        out = {"positions": frames, "materials": mats}
        if args.thermal:
            out["temps"] = temps
        np.savez_compressed(os.path.join(args.out, f"traj_{t:04d}.npz"), **out)
        kinds = ("thermal-pool" if args.thermal
                 else "pure-water" if pure
                 else "+".join(sorted({MATERIAL_NAMES[m] for m in set(mats.tolist())})))
        print(f"[gen] traj {t} ({kinds}): {args.frames} frames, "
              f"spacing={spacing:.5f} [{idx+1}/{len(plan)}]", flush=True)

    mean_spacing = float(np.mean(spacings))
    meta = {
        "n_particles": args.particles,
        "frames": args.frames,
        "dt_record": 1e-4 * args.substeps_per_frame,
        "mean_spacing": mean_spacing,
        "radius": round(2.5 * mean_spacing, 6),
        "domain": [0.0, 1.0],
        "wall_margin": 3.0 / n_grid,
        "arch": arch,
        "dim": 2,
        "gravity": 9.8,
        "n_materials": 0 if args.thermal else 3,
        "materials": MATERIAL_NAMES,
        "thermal": bool(args.thermal),
        "pure_water_holdout": 0 if args.thermal else args.pure_water_traj,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[gen] wrote meta.json: radius={meta['radius']}")


if __name__ == "__main__":
    main()
