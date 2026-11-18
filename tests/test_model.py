"""Model invariants: conservation by construction, chunking and checkpointing
must not change the math."""

import torch

from orrery import data as D
from orrery.model import GNS


def _graph(n=200, dim=2, node_in=14, radius=0.12):
    pos = torch.rand(n, dim)
    ei = D.build_edges(pos, radius)
    ef = D.edge_features(pos, ei, radius)
    nf = torch.randn(n, node_in)
    return nf, ei, ef


def test_param_count_at_defaults():
    m = GNS(node_in=14)
    n = sum(p.numel() for p in m.parameters())
    assert 1.0e6 < n < 2.5e6  # "~1-2M params": the world fills VRAM, not this


def test_conserve_momentum_is_exact_for_any_weights():
    nf, ei, ef = _graph()
    for seed in range(3):
        torch.manual_seed(seed)
        m = GNS(node_in=nf.shape[1], hidden=32, layers=3, conserve_momentum=True)
        out = m(nf, ei, ef)
        # every directed edge adds +g to dst and -g to src: the sum cancels
        assert out.sum(0).abs().max() < 1e-4 * out.abs().sum(0).max()


def test_free_decoder_does_not_conserve():
    nf, ei, ef = _graph()
    m = GNS(node_in=nf.shape[1], hidden=32, layers=3)
    out = m(nf, ei, ef)
    assert out.sum(0).abs().max() > 1e-3


def test_edge_chunked_inference_matches_full():
    nf, ei, ef = _graph(n=400)
    for conserve in (False, True):
        m = GNS(node_in=nf.shape[1], hidden=32, layers=3,
                conserve_momentum=conserve).eval()
        with torch.no_grad():
            full = m(nf, ei, ef)
            chunked = m(nf, ei, ef, edge_chunk=257)
        assert torch.allclose(full, chunked, atol=1e-5)


def test_grad_checkpoint_gives_identical_gradients():
    nf, ei, ef = _graph()
    tgt = torch.randn(nf.shape[0], 2)
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(1)
        m = GNS(node_in=nf.shape[1], hidden=32, layers=3, grad_checkpoint=ckpt)
        m.train()
        torch.nn.functional.mse_loss(m(nf, ei, ef), tgt).backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_material_and_regime_embeddings():
    nf, ei, ef = _graph()
    n = nf.shape[0]
    m = GNS(node_in=nf.shape[1], hidden=32, layers=2, out_dim=3,
            n_materials=3, n_regimes=2)
    out = m(nf, ei, ef, mat=torch.randint(0, 3, (n,)),
            regime=torch.randint(0, 2, (n,)))
    assert out.shape == (n, 3)
