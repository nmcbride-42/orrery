"""Neighbor search: the O(N) hash grid must return exactly the cdist graph."""

import pytest
import torch

from orrery import data as D


def _edge_set(ei):
    return set(map(tuple, ei.t().tolist()))


def _exact_edges(pos, radius):
    """Reference graph from direct differences (no matmul shortcut)."""
    d = (pos[:, None, :] - pos[None, :, :]).norm(dim=-1)
    hit = ((d < radius) & ~torch.eye(len(pos), dtype=torch.bool)).nonzero()
    return set(map(tuple, hit.tolist()))


def _near_boundary(pos, pairs, radius, tol=1e-5):
    return all(abs((pos[i] - pos[j]).norm().item() - radius) < tol
               for i, j in pairs)


@pytest.mark.parametrize("dim,n,radius", [(2, 3000, 0.03), (3, 2000, 0.08)])
def test_hash_grid_matches_exact_radius_graph(dim, n, radius):
    pos = torch.rand(n, dim)
    grid = _edge_set(D.build_edges_grid(pos, radius))
    assert grid == _exact_edges(pos, radius)
    # the small-N path uses torch.cdist, whose matmul formulation rounds
    # distances by ~1e-6: the two paths may disagree only on pairs sitting
    # within float error of the radius
    small = _edge_set(D.build_edges(pos, radius))
    assert _near_boundary(pos, grid ^ small, radius)
    assert len(grid ^ small) < 1e-3 * len(grid)


def test_hash_grid_handles_wide_domains():
    # big worlds are [0, W] x [0, 1] with W >> 1
    pos = torch.rand(4000, 2) * torch.tensor([20.0, 1.0])
    assert _edge_set(D.build_edges_grid(pos, 0.05)) == _exact_edges(pos, 0.05)


def test_edges_are_symmetric_and_self_free():
    pos = torch.rand(500, 2)
    e = _edge_set(D.build_edges(pos, 0.1))
    assert all((j, i) in e for i, j in e)
    assert all(i != j for i, j in e)


def test_edge_features():
    pos = torch.tensor([[0.0, 0.0], [0.03, 0.04]])
    ei = torch.tensor([[0, 1], [1, 0]])
    ef = D.edge_features(pos, ei, radius=0.1)
    assert ef.shape == (2, 3)
    assert torch.allclose(ef[:, 2], torch.tensor([0.5, 0.5]))
    assert torch.allclose(ef[0, :2], -ef[1, :2])
