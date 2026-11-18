"""Trainer details that the end-to-end pipeline test does not pin down."""

import os
import re

import torch

from orrery.train import is_oom
from conftest import run


def test_is_oom_classifies_without_raising():
    assert not is_oom(RuntimeError("shape mismatch"))
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert is_oom(RuntimeError("HIPBLAS_STATUS_ALLOC_FAILED"))
    oom_cls = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    assert is_oom(oom_cls("allocator gave up"))


def test_stop_at_keeps_the_steps_lr_horizon(nbody_run, tmp_path):
    data, _, _ = nbody_run
    out = str(tmp_path / "run")
    log = run("orrery.train", "--data", data, "--out", out,
              "--steps", "1000", "--stop-at", "5",
              "--batch", "1", "--holdout", "1", "--hidden", "32",
              "--layers", "2", "--noise", "auto", "--conserve-momentum",
              "--log-every", "5", "--device", "cpu")
    ck = torch.load(os.path.join(out, "latest.pt"), weights_only=False)
    assert ck["step"] == 5
    # LR at step 5 of a 1000-step decay is still ~1e-4; a 5-step horizon
    # would already have decayed to ~2.5e-6.
    lr = float(re.search(r"step 5/1000 .* lr=([0-9.e+-]+)", log).group(1))
    assert lr > 9e-5
