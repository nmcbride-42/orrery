import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    config.addinivalue_line("markers", "taichi: needs the optional taichi dependency")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def write_meta(path, **overrides):
    """Minimal 2D water-style corpus meta for tests."""
    meta = {
        "n_particles": 64,
        "frames": 16,
        "dt_record": 0.003,
        "radius": 0.08,
        "domain": [0.0, 1.0],
        "wall_margin": 0.02,
        "dim": 2,
        "walls": True,
        "gravity": 9.8,
    }
    meta.update(overrides)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return meta


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args):
    env = dict(os.environ, PYTHONPATH=ROOT)
    res = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


@pytest.fixture(scope="session")
def nbody_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("nbody")
    data, out = str(root / "data"), str(root / "run")
    run("orrery.gen.nbody", "--out", data, "--traj", "3", "--particles", "64",
        "--frames", "20", "--substeps-per-frame", "4", "--device", "cpu")
    log = run("orrery.train", "--data", data, "--out", out, "--steps", "20",
              "--batch", "1", "--holdout", "1", "--hidden", "32",
              "--layers", "2", "--noise", "auto", "--conserve-momentum",
              "--ckpt-every", "10", "--log-every", "10", "--device", "cpu")
    return data, out, log
