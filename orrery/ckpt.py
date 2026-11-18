"""Checkpoint loading shared by every inference entry point.

A checkpoint written by train.py carries the model weights, the
normalization stats, and a config dict describing the feature layout. This
module rebuilds the exact model from that config so rollout, whatif,
proptime, bigworld and export all load checkpoints the same way.
"""

import dataclasses

import torch

from . import data as D
from .model import GNS


def build_model(cfg: dict) -> GNS:
    """Instantiate a GNS whose shape matches a checkpoint config."""
    return GNS(
        node_in=cfg["node_in"],
        edge_in=cfg.get("edge_in", 3),
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        out_dim=cfg.get("out_dim", 2),
        n_materials=cfg.get("n_materials", 0),
        mat_dim=cfg.get("mat_dim", 16),
        n_regimes=cfg.get("n_regimes", 0),
        conserve_momentum=cfg.get("conserve_momentum", False),
    )


def load_checkpoint(path: str, device, regime: int = -1, verbose: bool = True):
    """-> (model in eval mode, cfg, stats, step).

    Merged (multi-corpus) checkpoints store per-regime stats as a list; the
    entry for `regime` is selected (regime 0 when none is given).
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg, stats = ck["config"], ck["stats"]
    if isinstance(stats, list):
        ri = regime if regime >= 0 else 0
        if regime < 0 and verbose:
            print("[ckpt] merged checkpoint with per-regime stats and no "
                  "--regime given; defaulting to regime 0")
        stats = stats[ri]
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg, stats, ck.get("step", -1)


def eval_meta(data_dir: str, cfg: dict) -> D.Meta:
    """Corpus meta with feature flags taken from the CHECKPOINT.

    Feature widths follow the checkpoint, not the eval corpus: a merged net
    evaluating a temperature-less corpus still expects its temperature and
    material channels.
    """
    return dataclasses.replace(
        D.load_meta(data_dir),
        thermal=cfg.get("thermal", False),
        n_materials=cfg.get("n_materials", 0),
        charged=cfg.get("charged", False),
    )
