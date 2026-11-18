"""Features, targets and batching."""

import numpy as np
import torch

from orrery import data as D
from orrery.spines import uniform_spine

from conftest import write_meta


def _freefall(n=50, frames=20, g_frame=1e-4):
    """Exact free fall in recorded units: a = -g_frame per frame^2."""
    rng = np.random.default_rng(0)
    p0 = rng.uniform(0.3, 0.7, (n, 2))
    v0 = rng.normal(0, 1e-3, (n, 2))
    t = np.arange(frames, dtype=np.float64)[:, None, None]
    pos = p0 + v0 * t
    pos[..., 1] -= 0.5 * g_frame * t[..., 0] ** 2
    return pos.astype(np.float32)


def test_spine_residual_is_zero_on_pure_gravity(tmp_path):
    write_meta(tmp_path)
    meta = D.load_meta(str(tmp_path))
    g = 1e-4
    traj = _freefall(g_frame=g)
    stats = {"vel_mean": [0, 0], "vel_std": [1, 1],
             "acc_mean": [0, 0], "acc_std": [1, 1]}
    _, _, _, target, _, _ = D.make_sample(traj, 10, stats, meta, "cpu",
                                          spine=uniform_spine(g, 2))
    assert target.abs().max() < 1e-6  # nothing left for the network to learn
    _, _, _, raw, _, _ = D.make_sample(traj, 10, stats, meta, "cpu")
    assert torch.allclose(raw[:, 1], torch.full((50,), -g), atol=1e-6)


def test_compute_stats_with_spine_removes_gravity():
    g = 1e-4
    s = D.compute_stats([_freefall(g_frame=g)], spine=uniform_spine(g, 2))
    assert abs(s["acc_mean"][1]) < 1e-6
    s_raw = D.compute_stats([_freefall(g_frame=g)])
    assert abs(s_raw["acc_mean"][1] + g) < 1e-6


def test_node_feature_width_matches_features(tmp_path):
    for flags in ({}, {"thermal": True}, {"charged": True},
                  {"walls": False}, {"dim": 3}):
        write_meta(tmp_path, **flags)
        meta = D.load_meta(str(tmp_path))
        dim, n = meta.dim, 30
        seq = torch.rand(D.HISTORY + 1, n, dim)
        stats = {"vel_mean": [0] * dim, "vel_std": [1] * dim,
                 "temp_mean": 0.5, "temp_std": 0.1}
        nf = D.node_features(
            seq, stats, meta, "cpu",
            temp_seq=torch.rand(D.HISTORY + 1, n) if meta.thermal else None,
            charge=torch.ones(n) if meta.charged else None)
        assert nf.shape == (n, D.node_feature_width(meta)), flags


def test_noise_leaves_first_frame_and_target_clean(tmp_path):
    write_meta(tmp_path)
    meta = D.load_meta(str(tmp_path))
    traj = _freefall()
    w = D._noisy_window(traj, 10, "cpu", noise_std=1e-2)
    clean = torch.tensor(traj[10 - D.HISTORY: 11])
    assert torch.equal(w[0], clean[0])
    assert not torch.allclose(w[-1], clean[-1])


def test_unrolled_sample_shapes(tmp_path):
    write_meta(tmp_path)
    meta = D.load_meta(str(tmp_path))
    stats = D.compute_stats([_freefall()])
    window, futures, temp, mat, reg, chg = D.make_sample(
        _freefall(), 8, stats, meta, "cpu", rollout_k=3, regime=1)
    assert window.shape == (D.HISTORY + 1, 50, 2)
    assert futures.shape == (3, 50, 2)
    assert temp is None and mat is None and chg is None
    assert torch.all(reg == 1)


def test_collate_offsets_edge_indices(tmp_path):
    write_meta(tmp_path)
    meta = D.load_meta(str(tmp_path))
    traj = _freefall()
    stats = D.compute_stats([traj])
    a = D.make_sample(traj, 8, stats, meta, "cpu")
    b = D.make_sample(traj, 9, stats, meta, "cpu")
    nf, ei, ef, tgt, mat, reg = D.collate([a, b])
    assert nf.shape[0] == 100 and tgt.shape[0] == 100
    assert ei.shape[1] == a[1].shape[1] + b[1].shape[1]
    assert ei[:, a[1].shape[1]:].min() >= 50
    assert mat is None and reg is None
