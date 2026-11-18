"""GNS-style encoder-processor-decoder in pure PyTorch.

No torch_geometric dependency: message aggregation is index_add_. The model
predicts normalized per-particle acceleration from a particle graph.
~1.6M parameters at the defaults — the world is what fills VRAM, not this.
"""

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: int, out_dim: int, layernorm: bool = True) -> nn.Module:
    layers = [
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    ]
    if layernorm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class GNBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = mlp(hidden * 3, hidden, hidden)
        self.node_mlp = mlp(hidden * 2, hidden, hidden)

    def forward(self, x, edge_index, e, edge_chunk: int = 0):
        src, dst = edge_index[0], edge_index[1]
        if edge_chunk and e.shape[0] > edge_chunk:
            # big-world inference path: never materialize all edge messages
            e_out = torch.empty_like(e)
            agg = torch.zeros_like(x)
            for s in range(0, e.shape[0], edge_chunk):
                t = min(s + edge_chunk, e.shape[0])
                e_new = self.edge_mlp(
                    torch.cat([e[s:t], x[src[s:t]], x[dst[s:t]]], dim=1)
                )
                agg.index_add_(0, dst[s:t], e_new)
                e_out[s:t] = e[s:t] + e_new
            x_new = self.node_mlp(torch.cat([x, agg], dim=1))
            return x + x_new, e_out
        e_new = self.edge_mlp(torch.cat([e, x[src], x[dst]], dim=1))
        agg = torch.zeros_like(x).index_add_(0, dst, e_new)
        x_new = self.node_mlp(torch.cat([x, agg], dim=1))
        return x + x_new, e + e_new


class GNS(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int = 3,
        hidden: int = 128,
        layers: int = 10,
        out_dim: int = 2,
        n_materials: int = 0,
        mat_dim: int = 16,
        n_regimes: int = 0,
        regime_dim: int = 8,
        conserve_momentum: bool = False,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        # material embedding = per-particle "which rulebook" tag; edges see
        # both endpoints' tags through message passing, so cross-material
        # physics is learnable (from mixed scenes only)
        self.mat_emb = nn.Embedding(n_materials, mat_dim) if n_materials else None
        # regime embedding = per-corpus "which teacher" tag (Gen 10 merge).
        # Required, not decorative: teachers legitimately disagree (thermal
        # water is E=400 + buoyancy, multi-material water is E=1000, no
        # buoyancy) and without the tag those are contradictory labels for
        # identical inputs.
        self.regime_emb = nn.Embedding(n_regimes, regime_dim) if n_regimes else None
        self.node_enc = mlp(
            node_in + (mat_dim if n_materials else 0)
            + (regime_dim if n_regimes else 0), hidden, hidden)
        self.edge_enc = mlp(edge_in, hidden, hidden)
        self.blocks = nn.ModuleList(GNBlock(hidden) for _ in range(layers))
        # conserve_momentum: Newton's third law by construction. The decoder
        # emits a force per DIRECTED edge, applied +g to dst and -g to src —
        # every edge's contribution cancels exactly, so net momentum change is
        # zero regardless of learned weights (injected law = one less thing to
        # hallucinate). Gen-4 lesson: the free-decoder student sank
        # its cloud's center of mass (-0.125/300 frames). Pair with corpora
        # whose residual acc_mean ~ 0 (true for nbody: symmetric dynamics).
        self.conserve = conserve_momentum  # decoder input: edge embedding if set
        # grad_checkpoint: recompute block activations in backward instead of
        # retaining them — ~3-5x less activation memory for ~30% speed. Dense
        # graph frames (300-500k edges) OOM a 16GB card at k=1 otherwise, and
        # skipping them censors exactly the contact physics.
        self.grad_ckpt = grad_checkpoint
        self.decoder = mlp(hidden, hidden, out_dim, layernorm=False)

    def forward(self, node_feats, edge_index, edge_feats, edge_chunk: int = 0,
                mat=None, regime=None):
        if self.mat_emb is not None:
            node_feats = torch.cat([node_feats, self.mat_emb(mat)], dim=1)
        if self.regime_emb is not None:
            node_feats = torch.cat([node_feats, self.regime_emb(regime)], dim=1)
        if edge_chunk and edge_feats.shape[0] > edge_chunk:
            e = torch.empty(
                edge_feats.shape[0], self.edge_enc[-1].normalized_shape[0]
                if hasattr(self.edge_enc[-1], "normalized_shape")
                else self.edge_enc[-3].out_features,
                dtype=node_feats.dtype, device=node_feats.device,
            )
            for s in range(0, edge_feats.shape[0], edge_chunk):
                t = min(s + edge_chunk, edge_feats.shape[0])
                e[s:t] = self.edge_enc(edge_feats[s:t])
        else:
            e = self.edge_enc(edge_feats)
        x = self.node_enc(node_feats)
        use_ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                x, e = torch.utils.checkpoint.checkpoint(
                    block, x, edge_index, e, use_reentrant=False)
            else:
                x, e = block(x, edge_index, e, edge_chunk=edge_chunk)
        if not self.conserve:
            return self.decoder(x)
        src, dst = edge_index[0], edge_index[1]
        out_dim = self.decoder[-1].out_features
        acc = x.new_zeros(x.shape[0], out_dim)
        if edge_chunk and e.shape[0] > edge_chunk:
            for s in range(0, e.shape[0], edge_chunk):
                t = min(s + edge_chunk, e.shape[0])
                g = self.decoder(e[s:t])
                acc.index_add_(0, dst[s:t], g)
                acc.index_add_(0, src[s:t], -g)
        else:
            g = self.decoder(e)
            acc.index_add_(0, dst, g)
            acc.index_add_(0, src, -g)
        return acc
