"""Exact-force spines: analytically known accelerations the network never learns.

Every injected law is one less thing the network can hallucinate (docs/design.md).
A spine is a callable pos [N,D] -> acc [N,D] in recorded-frame units
(displacement per frame^2). The trainer subtracts it from the target; the
rollout adds it back at every step.

  uniform: constant gravity along -y (Gen 2/3/5 — the Honest Water spine)
  nbody:   softened pairwise self-gravity, equal masses (Gen 4 — the Heavens)
"""

import torch


def uniform_spine(g_frame: float, dim: int):
    """Constant gravity; g_frame = g * dt_record^2 (recorded units)."""
    def spine(pos: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros_like(pos)
        acc[:, 1] = -g_frame
        return acc
    return spine


def nbody_spine(g_frame: float, soften: float, chunk: int = 4096):
    """Softened pairwise gravity, equal unit masses.

    g_frame = G * dt_record^2 (recorded units). Chunked over sources so the
    [N,N] pair tensor never exceeds chunk*N (100k-body worlds stay feasible).
    """
    def spine(pos: torch.Tensor) -> torch.Tensor:
        n = pos.shape[0]
        acc = torch.zeros_like(pos)
        eps2 = soften * soften
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            rel = pos.unsqueeze(0) - pos[s:e].unsqueeze(1)   # [c, N, D]
            d2 = (rel * rel).sum(-1) + eps2                  # [c, N]
            inv = d2.pow(-1.5)
            inv[:, s:e].fill_diagonal_(0.0)                  # no self-force
            acc[s:e] = g_frame * (rel * inv.unsqueeze(-1)).sum(1)
        return acc
    return spine


def coulomb_spine(k_frame: float, soften: float, trap_w2_frame: float,
                  charges: torch.Tensor, chunk: int = 4096):
    """Signed Coulomb + harmonic trap (Gen 9). Per-TRAJECTORY: the charge
    pattern varies, so callers build one spine per trajectory.

    k_frame = K * dt_record^2, trap_w2_frame = trap_w2 * dt_record^2.
    """
    def spine(pos: torch.Tensor) -> torch.Tensor:
        n = pos.shape[0]
        q = charges.to(pos.device)
        acc = -trap_w2_frame * (pos - 0.5)
        eps2 = soften * soften
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            rel = pos[s:e].unsqueeze(1) - pos.unsqueeze(0)
            d2 = (rel * rel).sum(-1) + eps2
            inv = d2.pow(-1.5)
            inv[:, s:e].fill_diagonal_(0.0)
            qq = q[s:e].unsqueeze(1) * q.unsqueeze(0)
            acc[s:e] += k_frame * (qq.unsqueeze(-1) * rel * inv.unsqueeze(-1)).sum(1)
        return acc
    return spine


def make_spine(config: dict, charges=None):
    """Build a spine from a ckpt/meta config dict; None if no spine."""
    kind = config.get("spine", "uniform" if config.get("g_frame") else None)
    if kind in (None, "none"):
        return None
    if kind == "uniform":
        return uniform_spine(config["g_frame"], config.get("dim", 2))
    if kind == "nbody":
        return nbody_spine(config["g_frame"], config["soften"])
    if kind == "coulomb":
        assert charges is not None, "coulomb spine needs the trajectory's charges"
        return coulomb_spine(config["g_frame"], config["soften"],
                             config["trap_w2_frame"], charges)
    raise ValueError(f"unknown spine kind: {kind}")
