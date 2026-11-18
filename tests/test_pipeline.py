"""End to end on CPU: teacher -> corpus -> train -> checkpoint -> rollout.

Tiny sizes; this checks the plumbing, not the physics quality.
"""

import json
import os

import numpy as np
import pytest
import torch

from orrery import data as D
from orrery.ckpt import eval_meta, load_checkpoint
from orrery.proptime import gated_rollout
from orrery.rollout import rollout
from orrery.whatif import SCENES, imagine, seed_window

from conftest import run

def test_nbody_teacher_writes_corpus(nbody_run):
    data, _, _ = nbody_run
    meta = json.load(open(os.path.join(data, "meta.json")))
    assert meta["G"] > 0 and meta["walls"] is False
    trajs = D.load_corpus(data)
    assert len(trajs) == 3 and trajs[0].pos.shape == (20, 64, 2)
    assert all(np.isfinite(t.pos).all() for t in trajs)


def test_train_uses_nbody_spine_and_checkpoints(nbody_run):
    _, out, log = nbody_run
    assert "spine=nbody" in log
    assert "step 20/20" in log
    ck = torch.load(os.path.join(out, "latest.pt"), weights_only=False)
    assert ck["step"] == 20 and ck["config"]["conserve_momentum"]


def test_resume_continues_from_checkpoint(nbody_run):
    data, out, _ = nbody_run
    log = run("orrery.train", "--data", data, "--out", out, "--steps", "30",
              "--batch", "1", "--holdout", "1", "--hidden", "32",
              "--layers", "2", "--noise", "auto", "--conserve-momentum",
              "--ckpt-every", "10", "--log-every", "10", "--device", "cpu",
              "--resume")
    assert "resumed from step 20" in log and "step 30/30" in log


def test_rollout_cli_and_gif(nbody_run, tmp_path):
    data, out, _ = nbody_run
    gif = str(tmp_path / "r.gif")
    log = run("orrery.rollout", "--ckpt", os.path.join(out, "latest.pt"),
              "--data", data, "--steps", "10", "--metrics", "--gif", gif,
              "--device", "cpu")
    assert "full-rollout position MSE" in log
    assert os.path.getsize(gif) > 0


def test_conserving_rollout_com_drift_bounded_by_mean_residual(nbody_run):
    """The network term conserves momentum exactly; the only com motion left
    comes from denormalization's corpus-mean residual acceleration."""
    data, out, _ = nbody_run
    model, cfg, stats, _ = load_checkpoint(os.path.join(out, "latest.pt"), "cpu")
    meta = eval_meta(data, cfg)
    truth = D.load_corpus(data)[-1]
    steps = 12
    pred, _ = rollout(model, truth, stats, meta, cfg, "cpu", steps)
    com = pred.astype(np.float64).mean(1)
    seed = D.HISTORY + 1
    v0 = com[seed - 1] - com[seed - 2]
    # com of a momentum-conserving system moves at constant velocity, plus
    # 0.5 * k^2 * acc_mean under a constant per-step bias
    k = np.arange(1, steps + 1)[:, None]
    expect = com[seed - 1] + v0 * k + 0.5 * k * (k + 1) * np.array(stats["acc_mean"])
    assert np.abs(com[seed:] - expect).max() < 1e-5


def test_property_time_all_hot_is_bit_exact(nbody_run):
    """With every particle hot, gating must not change a single bit."""
    data, out, _ = nbody_run
    model, cfg, stats, _ = load_checkpoint(os.path.join(out, "latest.pt"), "cpu")
    meta = eval_meta(data, cfg)
    truth = D.load_corpus(data)[0]
    full, _ = rollout(model, truth, stats, meta, cfg, "cpu", 8)
    gated, info = gated_rollout(model, truth, stats, meta, cfg, "cpu", 8,
                                v_frac=0.0, coarse_every=1000)
    assert info["node_updates"] == info["node_updates_full"]
    assert np.array_equal(full, gated)


def test_property_time_freezes_settled_particles(nbody_run):
    data, out, _ = nbody_run
    model, cfg, stats, _ = load_checkpoint(os.path.join(out, "latest.pt"), "cpu")
    meta = eval_meta(data, cfg)
    truth = D.load_corpus(data)[0]
    _, info = gated_rollout(model, truth, stats, meta, cfg, "cpu", 8,
                            v_frac=1e6, coarse_every=4, dwell=0)
    # only the full-sync ticks (steps 0 and 4) update anyone
    assert info["node_updates"] == 2 * truth.pos.shape[1]


def test_whatif_scene_from_untrained_water_model(tmp_path):
    """whatif/export plumbing on a randomly initialised water-config model."""
    from conftest import write_meta
    from orrery.ckpt import build_model

    write_meta(tmp_path, radius=0.03)
    cfg = {"node_in": 2 * D.HISTORY + 4, "edge_in": 3, "out_dim": 2,
           "hidden": 16, "layers": 1, "dim": 2, "walls": True,
           "spine": "uniform", "g_frame": 9.8 * 0.003 ** 2, "wall_clip": 4.0}
    model = build_model(cfg).eval()
    stats = {"vel_mean": [0, 0], "vel_std": [1e-3, 1e-3],
             "acc_mean": [0, 0], "acc_std": [1e-5, 1e-5]}
    meta = eval_meta(str(tmp_path), cfg)
    assert seed_window(SCENES["droplet"](np.random.default_rng(0))).shape == \
        (D.HISTORY + 1, 2700, 2)
    pred = imagine(model, cfg, stats, meta, "two_blocks", 3, "cpu")
    assert pred.shape == (D.HISTORY + 1 + 3, 3000, 2)
    assert np.isfinite(pred).all()


@pytest.mark.taichi
def test_mpm2d_water_teacher_cpu(tmp_path):
    pytest.importorskip("taichi")
    data = str(tmp_path / "water")
    run("orrery.gen.mpm2d", "--out", data, "--traj", "2", "--particles", "200",
        "--frames", "10", "--arch", "cpu")
    meta = D.load_meta(data)
    assert meta.radius > 0 and meta.walls
    trajs = D.load_corpus(data)
    assert len(trajs) == 2 and trajs[0].pos.shape == (10, 200, 2)
    pos = trajs[0].pos
    assert pos.min() >= 0.0 and pos.max() <= 1.0


def test_bigworld_fp16_keeps_state_fp32(tmp_path):
    """--fp16 halves the network only; positions and the integrator stay fp32."""
    from conftest import write_meta
    from orrery.ckpt import build_model

    write_meta(tmp_path, radius=0.02)
    cfg = {"node_in": 2 * D.HISTORY + 4, "edge_in": 3, "out_dim": 2,
           "hidden": 16, "layers": 1, "dim": 2, "walls": True,
           "spine": "uniform", "g_frame": 9.8 * 0.003 ** 2, "wall_clip": 4.0}
    stats = {"vel_mean": [0, 0], "vel_std": [1e-3, 1e-3],
             "acc_mean": [0, 0], "acc_std": [1e-5, 1e-5]}
    ckpt = str(tmp_path / "w.pt")
    torch.save({"model": build_model(cfg).state_dict(), "config": cfg,
                "stats": stats, "step": 0}, ckpt)
    log = run("orrery.bigworld", "--ckpt", ckpt, "--data", str(tmp_path),
              "--particles", "4000", "--steps", "3", "--fp16", "--device", "cpu")
    assert "model dtype=torch.float16, state dtype=torch.float32" in log
    assert "NaN" not in log and "edges" in log
