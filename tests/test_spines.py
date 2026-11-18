"""Exact-force spines must be the SAME law the teachers integrate."""

import torch

from orrery.gen import charged, nbody
from orrery.spines import (coulomb_spine, make_spine, nbody_spine,
                           uniform_spine)


def test_uniform_spine():
    acc = uniform_spine(0.01, 2)(torch.rand(10, 2))
    assert torch.all(acc[:, 0] == 0) and torch.all(acc[:, 1] == -0.01)


def test_nbody_spine_matches_teacher_gravity():
    pos = 0.3 + 0.4 * torch.rand(300, 2)
    vel = torch.zeros_like(pos)
    G, soften = 1.3e-4, 0.004
    # r_p = 0 disables the teacher's contact term, leaving pure gravity
    teacher = nbody.pair_forces(pos, vel, G, soften, r_p=0.0, k_n=0.0, c_n=0.0)
    spine = nbody_spine(G, soften, chunk=64)(pos)
    assert torch.allclose(teacher, spine, rtol=1e-4, atol=1e-7)


def test_nbody_spine_conserves_momentum():
    acc = nbody_spine(1e-4, 0.004, chunk=50)(torch.rand(400, 2))
    assert acc.sum(0).abs().max() < 1e-4 * acc.abs().sum(0).max()


def test_coulomb_spine_matches_teacher():
    n = 200
    pos = 0.3 + 0.4 * torch.rand(n, 2)
    q = torch.ones(n)
    q[torch.randperm(n)[: n // 2]] = -1.0
    K, soften, w2 = 1.3e-4, 0.004, 6.0
    teacher = charged.pair_forces(pos, torch.zeros_like(pos), q, K, soften,
                                  r_p=0.0, k_n=0.0, c_n=0.0, trap_w2=w2)
    spine = coulomb_spine(K, soften, w2, q, chunk=64)(pos)
    assert torch.allclose(teacher, spine, rtol=1e-4, atol=1e-7)


def test_make_spine_dispatch():
    assert make_spine({}) is None
    assert make_spine({"spine": "none"}) is None
    assert make_spine({"g_frame": 1e-4}) is not None  # legacy uniform
    assert make_spine({"spine": "nbody", "g_frame": 1e-4, "soften": 0.01})
