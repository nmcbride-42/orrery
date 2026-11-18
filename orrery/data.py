"""Dataset, graph building, normalization, and noise injection.

Feature scheme (GNS-style):
  node: C recent velocities (normalized) + distance to 4 walls (clipped to R, /R)
  edge: relative displacement / R (2) + distance / R (1)
  target: acceleration p[t+1] - 2p[t] + p[t-1], normalized

Recorded-frame dt is folded into the units (velocity = per-frame displacement),
so the integrator at rollout is p_next = 2p - p_prev + acc with no explicit dt.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

HISTORY = 5  # velocities of context per particle


@dataclass
class Meta:
    radius: float
    n_particles: int
    frames: int
    domain: tuple
    wall_margin: float
    dt_record: float = 0.0
    dim: int = 2
    walls: bool = True
    gravity: float = 9.8
    n_materials: int = 0
    thermal: bool = False
    # nbody corpora: G and softening in physical units (see gen/nbody.py)
    G: float = 0.0
    soften: float = 0.0
    # charged corpora (gen/charged.py): Coulomb K + harmonic trap
    charged: bool = False
    K: float = 0.0
    trap_w2: float = 0.0


def load_meta(data_dir: str) -> Meta:
    with open(os.path.join(data_dir, "meta.json")) as fh:
        m = json.load(fh)
    return Meta(
        radius=m["radius"],
        n_particles=m["n_particles"],
        frames=m["frames"],
        domain=tuple(m["domain"]),
        wall_margin=m.get("wall_margin", 0.0),
        dt_record=m.get("dt_record", 0.0),
        dim=m.get("dim", 2),
        walls=m.get("walls", True),
        gravity=m.get("gravity", 9.8),
        n_materials=m.get("n_materials", 0),
        thermal=m.get("thermal", False),
        G=m.get("G", 0.0),
        soften=m.get("soften", 0.0),
        charged=m.get("charged", False),
        K=m.get("K", 0.0),
        trap_w2=m.get("trap_w2", 0.0),
    )


def gravity_per_frame(meta: Meta) -> float:
    """Exact gravity in recorded units: g m/s^2 * dt_record^2 (per frame^2)."""
    return meta.gravity * meta.dt_record ** 2


@dataclass
class Traj:
    pos: np.ndarray                  # [T, N, D]
    mat: np.ndarray | None = None    # [N] uint8
    temp: np.ndarray | None = None   # [T, N] float32
    charge: np.ndarray | None = None  # [N] float32 (+1/-1)

    @property
    def shape(self):
        return self.pos.shape

    def __len__(self):
        return self.pos.shape[0]


def load_corpus(data_dir: str) -> list[Traj]:
    files = sorted(
        f for f in os.listdir(data_dir) if f.startswith("traj_") and f.endswith(".npz")
    )
    out = []
    for f in files:
        d = np.load(os.path.join(data_dir, f))
        out.append(Traj(
            pos=d["positions"],
            mat=d["materials"] if "materials" in d else None,
            temp=d["temps"] if "temps" in d else None,
            charge=d["charges"] if "charges" in d else None,
        ))
    return out


def compute_stats(trajs: list, spine=None, thermal: bool = False,
                  device: str = "cpu", spine_factory=None) -> dict:
    """Per-component mean/std of velocities and accelerations across the corpus.

    spine (see spines.py): stats describe the acceleration RESIDUAL after the
    exact spine is removed — the only part the network learns. Accepts either
    raw position arrays or Traj objects. spine_factory (per-trajectory
    spines, e.g. coulomb) takes precedence over spine.
    """
    vels, accs, dts = [], [], []
    for tr in trajs:
        p = tr.pos if isinstance(tr, Traj) else tr
        spine_fn = spine_factory(tr) if spine_factory is not None else spine
        dim = p.shape[-1]
        v = p[1:] - p[:-1]
        a = v[1:] - v[:-1]
        if spine_fn is not None:
            # a[k] targets frame k+1: subtract the spine evaluated there
            pt = torch.tensor(p[1:-1].reshape(-1, dim), dtype=torch.float32,
                              device=device)
            n = p.shape[1]
            sp = torch.stack(
                [spine_fn(pt[i * n:(i + 1) * n]) for i in range(p.shape[0] - 2)]
            ).cpu().numpy()
            a = a - sp
        vels.append(v.reshape(-1, dim))
        accs.append(a.reshape(-1, dim))
        if thermal and isinstance(tr, Traj) and tr.temp is not None:
            dts.append((tr.temp[1:] - tr.temp[:-1]).reshape(-1))
    v = np.concatenate(vels)
    a = np.concatenate(accs)
    floor = 1e-7  # degenerate components (e.g. zero accel on an axis) must not explode normalization
    stats = {
        "vel_mean": v.mean(0).tolist(),
        "vel_std": np.maximum(v.std(0), floor).tolist(),
        "acc_mean": a.mean(0).tolist(),
        "acc_std": np.maximum(a.std(0), floor).tolist(),
    }
    if dts:
        dt_all = np.concatenate(dts)
        t_all = np.concatenate([tr.temp.reshape(-1) for tr in trajs
                                if isinstance(tr, Traj) and tr.temp is not None])
        stats["temp_mean"] = float(t_all.mean())
        stats["temp_std"] = float(max(t_all.std(), floor))
        stats["dtemp_mean"] = float(dt_all.mean())
        stats["dtemp_std"] = float(max(dt_all.std(), floor))
    return stats


def build_edges(pos: torch.Tensor, radius: float, chunk: int = 2048):
    """Radius graph. Small N: chunked cdist. Large N: hash grid (O(N)).

    Returns edge_index [2,E] (src, dst) excluding self-edges. Directed both
    ways (i->j and j->i both present) as GNS expects.
    """
    if pos.shape[0] > 20_000:
        return build_edges_grid(pos, radius)
    n = pos.shape[0]
    srcs, dsts = [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = torch.cdist(pos[start:end], pos)  # [c, N]
        hit = (d < radius).nonzero(as_tuple=False)
        src = hit[:, 0] + start
        dst = hit[:, 1]
        keep = src != dst
        srcs.append(src[keep])
        dsts.append(dst[keep])
    return torch.stack([torch.cat(srcs), torch.cat(dsts)])


def build_edges_grid(pos: torch.Tensor, radius: float, chunk: int = 200_000):
    """Hash-grid radius graph for large N. Cells of size `radius`; each
    particle checks the 3^D cell neighborhood. Memory scales with N, not N²."""
    device = pos.device
    n, dim = pos.shape
    cell = torch.div(pos, radius, rounding_mode="floor").long()  # [N,D]
    span = cell.max(0).values - cell.min(0).values + 3
    cell = cell - cell.min(0).values + 1  # pad so stencil offsets never go negative
    # row-major strides over the padded span
    strides = torch.ones(dim, dtype=torch.long, device=device)
    for d in range(dim - 2, -1, -1):
        strides[d] = strides[d + 1] * span[d + 1]
    key = (cell * strides).sum(1)
    order = torch.argsort(key)
    key_sorted = key[order]

    import itertools
    offsets = torch.tensor(
        list(itertools.product((-1, 0, 1), repeat=dim)), device=device
    )
    srcs, dsts = [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        idx = torch.arange(start, end, device=device)
        p_cell = cell[start:end]
        for off in offsets:
            nkey = ((p_cell + off) * strides).sum(1)
            lo = torch.searchsorted(key_sorted, nkey)
            hi = torch.searchsorted(key_sorted, nkey, right=True)
            counts = hi - lo
            if counts.max() == 0:
                continue
            src = idx.repeat_interleave(counts)
            ranges = torch.arange(int(counts.max()), device=device)
            cand = lo.unsqueeze(1) + ranges.unsqueeze(0)  # [c, maxc]
            valid = ranges.unsqueeze(0) < counts.unsqueeze(1)
            dst = order[cand[valid]]
            d = (pos[src] - pos[dst]).norm(dim=1)
            keep = (d < radius) & (src != dst)
            srcs.append(src[keep])
            dsts.append(dst[keep])
    return torch.stack([torch.cat(srcs), torch.cat(dsts)])


def node_features(
    pos_seq: torch.Tensor, stats: dict, meta: Meta, device, bounds=None,
    wall_clip: float = 1.0, temp_seq: torch.Tensor = None,
    charge: torch.Tensor = None,
) -> torch.Tensor:
    """pos_seq [C+1, N, D] (oldest..newest) -> [N, D*C (+ 2D walls) (+ C+1 temps)].

    bounds: optional (lo, hi) per axis, flattened (x_lo, x_hi, y_lo, y_hi, ...)
    for non-square domains (big worlds). Defaults to the square meta.domain.
    wall_clip: wall visibility range as a multiple of R (Gen 2 uses 4.0 —
    the phantom-floor fix; Gen 1 checkpoints trained with 1.0).
    temp_seq [C+1, N] (thermal corpora): normalized temperature history.
    """
    dtype = pos_seq.dtype
    dim = pos_seq.shape[-1]
    vel = pos_seq[1:] - pos_seq[:-1]  # [C, N, D]
    vm = torch.tensor(stats["vel_mean"], dtype=dtype, device=device)
    vs = torch.tensor(stats["vel_std"], dtype=dtype, device=device)
    vel = (vel - vm) / vs
    vel = vel.permute(1, 0, 2).reshape(pos_seq.shape[1], -1)  # [N, D*C]
    feats = [vel]
    if meta.walls:
        p = pos_seq[-1]
        if bounds is None:
            lo, hi = meta.domain
            bounds = (lo, hi) * dim
        reach = meta.radius * wall_clip
        cols = []
        for d in range(dim):  # interleaved (lo_d, hi_d) — Gen-2 ckpt order
            cols.append(p[:, d] - bounds[2 * d])
            cols.append(bounds[2 * d + 1] - p[:, d])
        walls = torch.stack(cols, dim=1)
        feats.append(torch.clamp(walls, max=reach) / reach)
    if temp_seq is not None:
        tm, ts = stats["temp_mean"], stats["temp_std"]
        feats.append(((temp_seq - tm) / ts).permute(1, 0))  # [N, C+1]
    if charge is not None:
        feats.append(charge.unsqueeze(1))  # [N, 1], already +/-1
    return torch.cat(feats, dim=1)


def node_feature_width(meta: Meta) -> int:
    """Input width implied by the corpus meta (must match training)."""
    w = meta.dim * HISTORY
    if meta.walls:
        w += 2 * meta.dim
    if meta.thermal:
        w += HISTORY + 1
    if meta.charged:
        w += 1
    return w


def edge_features(pos: torch.Tensor, edge_index: torch.Tensor, radius: float):
    """[E,3]: relative displacement / R and distance / R."""
    rel = (pos[edge_index[0]] - pos[edge_index[1]]) / radius
    dist = rel.norm(dim=1, keepdim=True)
    return torch.cat([rel, dist], dim=1)


def _noisy_window(pos: np.ndarray, t: int, device, noise_std: float):
    """[C+1, N, D] window ending at frame t, with random-walk noise."""
    window = torch.tensor(pos[t - HISTORY: t + 1], dtype=torch.float32, device=device)
    if noise_std > 0:
        step = noise_std / np.sqrt(HISTORY)
        noise = torch.randn_like(window) * step
        noise[0] = 0.0
        window = window + noise.cumsum(0)
    return window


def make_sample(
    traj,
    t: int,
    stats: dict,
    meta: Meta,
    device,
    noise_std: float = 0.0,
    g_frame: float = 0.0,
    wall_clip: float = 1.0,
    spine=None,
    rollout_k: int = 1,
    regime: int = None,
    synth_temp: float = 0.5,
):
    """One training sample at frame t. Needs t-HISTORY..t+rollout_k valid.

    Random-walk noise perturbs the input position window; the target is
    computed against the *clean* future positions so the model learns to
    correct drift (GNS recipe). A spine (or legacy g_frame) is removed from
    the target — the network learns the residual only.

    rollout_k == 1 returns (nf, ei, ef, target, mat, reg) — the one-step
    sample. rollout_k > 1 returns (window, futures, temps, mat, reg) for the
    unrolled loss (graphs rebuilt per unroll step inside the training loop).

    Merged-corpus mode (Gen 10): regime tags every node with the corpus id;
    a thermal=True meta over a temp-less corpus synthesizes a constant
    isothermal window at synth_temp = the teacher's t_ref, so "no buoyancy"
    is CONSISTENT physics (T == t_ref => zero buoyant force), not a
    contradiction.

    Accepts a Traj or a bare positions array.
    """
    pos = traj.pos if isinstance(traj, Traj) else traj
    mat = temp = reg = chg = None
    if regime is not None:
        reg = torch.full((pos.shape[1],), regime, dtype=torch.long, device=device)
    if isinstance(traj, Traj):
        if meta.charged and traj.charge is not None:
            chg = torch.tensor(traj.charge, dtype=torch.float32, device=device)
        if meta.n_materials:
            m = traj.mat if traj.mat is not None else np.zeros(pos.shape[1], np.uint8)
            mat = torch.tensor(m, dtype=torch.long, device=device)
        if meta.thermal:
            if traj.temp is not None:
                temp = torch.tensor(traj.temp[t - HISTORY: t + rollout_k + 1],
                                    dtype=torch.float32, device=device)
            else:
                temp = torch.full((HISTORY + rollout_k + 1, pos.shape[1]),
                                  synth_temp, dtype=torch.float32, device=device)
    if spine is None and g_frame > 0:
        from .spines import uniform_spine
        spine = uniform_spine(g_frame, meta.dim)

    window = _noisy_window(pos, t, device, noise_std)

    if rollout_k > 1:
        futures = torch.tensor(pos[t + 1: t + rollout_k + 1],
                               dtype=torch.float32, device=device)
        return window, futures, temp, mat, reg, chg

    nxt = torch.tensor(pos[t + 1], dtype=torch.float32, device=device)
    p_now, p_prev = window[-1], window[-2]
    acc = nxt - 2 * p_now + p_prev
    if spine is not None:
        acc = acc - spine(p_now)
    am = torch.tensor(stats["acc_mean"], dtype=torch.float32, device=device)
    astd = torch.tensor(stats["acc_std"], dtype=torch.float32, device=device)
    target = (acc - am) / astd
    if temp is not None:
        d_t = (temp[HISTORY + 1] - temp[HISTORY] - stats["dtemp_mean"]) / stats["dtemp_std"]
        target = torch.cat([target, d_t.unsqueeze(1)], dim=1)

    edge_index = build_edges(p_now, meta.radius)
    nf = node_features(window, stats, meta, device, wall_clip=wall_clip,
                       temp_seq=temp[: HISTORY + 1] if temp is not None else None,
                       charge=chg)
    ef = edge_features(p_now, edge_index, meta.radius)
    return nf, edge_index, ef, target, mat, reg


def collate(samples):
    """Concatenate one-step graphs with node-index offsets.

    Accepts legacy 4-tuples through 6-tuples (material, regime columns);
    absent columns come back as None.
    """
    nfs, eis, efs, tgts, mats, regs = [], [], [], [], [], []
    offset = 0
    for s in samples:
        nf, ei, ef, tgt = s[:4]
        mat = s[4] if len(s) > 4 else None
        reg = s[5] if len(s) > 5 else None
        nfs.append(nf)
        eis.append(ei + offset)
        efs.append(ef)
        tgts.append(tgt)
        if mat is not None:
            mats.append(mat)
        if reg is not None:
            regs.append(reg)
        offset += nf.shape[0]
    return (
        torch.cat(nfs),
        torch.cat(eis, dim=1),
        torch.cat(efs),
        torch.cat(tgts),
        torch.cat(mats) if mats else None,
        torch.cat(regs) if regs else None,
    )
