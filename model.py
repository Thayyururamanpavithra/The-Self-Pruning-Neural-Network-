"""
Self-Pruning Neural Network — model definitions.

Implements a custom `PrunableLinear` layer whose weights are element-wise
multiplied by learnable sigmoid "gates". An L1 penalty on the gates during
training drives many of them toward zero, effectively pruning connections.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """A linear layer with a learnable per-weight gate.

    Forward pass: y = (W * sigmoid(gate_scores)) @ x + b

    Both `weight` and `gate_scores` are `nn.Parameter`s, so autograd will
    propagate gradients through both during .backward().
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight, Kaiming-uniform init (same as nn.Linear).
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Gate scores init: normal distribution centered at -2 (sigmoid ≈ 0.12)
        # with large spread (std=2). Centering below zero puts most gates
        # in "slightly closed" state, so the classification loss needs to
        # actively open the useful ones while the L1 penalty keeps the
        # others closed. The large spread (std=2) is CRITICAL — it means
        # at init, ~16% of gates are in each tail (above 0 → ~sigmoid 0.5+
        # "open-ish", and below -4 → "very closed"). This gives the CE
        # gradient a differentiated starting point: some gates already
        # contribute to logits more than others, so CE can quickly figure
        # out which ones matter. Without this spread, all gates move in
        # lockstep under the uniform L1 gradient and pruning collapses.
        self.gate_scores = nn.Parameter(torch.empty_like(self.weight))
        nn.init.normal_(self.gate_scores, mean=-2.0, std=2.0)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)          # in (0, 1)
        pruned_weights = self.weight * gates             # element-wise
        return F.linear(x, pruned_weights, self.bias)

    # ------------------------------------------------------------------ #
    # Introspection helpers (used by training/eval and the API)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def gates(self) -> torch.Tensor:
        """Current gate values in (0, 1), detached."""
        return torch.sigmoid(self.gate_scores).detach()

    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> Tuple[int, int]:
        """Return (num_pruned, total) — a gate is 'pruned' if below threshold."""
        g = self.gates()
        total = g.numel()
        pruned = int((g < threshold).sum().item())
        return pruned, total

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class SelfPruningNet(nn.Module):
    """A feed-forward classifier built from PrunableLinear layers.

    Input: CIFAR-10 image flattened to 3072 (3*32*32).
    Hidden layers: [512, 256, 128] with ReLU.
    Output: 10 class logits.
    """

    def __init__(
        self,
        in_features: int = 3 * 32 * 32,
        hidden: Tuple[int, ...] = (512, 256, 128),
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [in_features, *hidden, num_classes]
        self.layers: nn.ModuleList = nn.ModuleList(
            [PrunableLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:        # no activation after logits
                x = F.relu(x)
                x = self.dropout(x)
        return x

    # ------------------------------------------------------------------ #
    # Sparsity loss & statistics
    # ------------------------------------------------------------------ #
    def sparsity_loss(self, form: str = "sigmoid",
                      margin: float = 6.0) -> torch.Tensor:
        """L1 sparsity penalty on the gates.

        Two forms:
          - "sigmoid" (the spec-literal one, the default):
                L = Σ sigmoid(gate_scores)
            This is the pure L1 norm of the gate outputs. Since gate values
            are in (0, 1), their L1 equals their sum. The gradient of this
            term w.r.t. gate_scores[i] is sigmoid(s)(1-sigmoid(s)), which
            is maximal (0.25) at s=0 and shrinks toward the tails. This
            means pressure is strongest on partially-open gates and weakest
            on already-committed (very open or very closed) ones — exactly
            the behavior we want.

          - "relu_shift" (alternative):
                L = Σ relu(gate_scores + margin)
            A straight-through style penalty. Each gate_score above -margin
            gets penalized linearly; once it drops below -margin, both loss
            and gradient become zero. Useful if you want faster,
            more decisive pruning at the cost of less smooth optimization.
        """
        total = torch.tensor(0.0, device=self.layers[0].weight.device)
        if form == "sigmoid":
            for layer in self.prunable_layers():
                total = total + torch.sigmoid(layer.gate_scores).sum()
        elif form == "relu_shift":
            for layer in self.prunable_layers():
                total = total + torch.relu(layer.gate_scores + margin).sum()
        else:
            raise ValueError(f"Unknown sparsity loss form: {form}")
        return total

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    @torch.no_grad()
    def sparsity_stats(self, threshold: float = 1e-2) -> dict:
        """Per-layer and global sparsity levels."""
        per_layer = []
        total_pruned = 0
        total_gates = 0
        for idx, layer in enumerate(self.prunable_layers()):
            pruned, total = layer.sparsity(threshold)
            per_layer.append({
                "layer": idx,
                "shape": list(layer.weight.shape),
                "total_weights": total,
                "pruned": pruned,
                "sparsity_pct": 100.0 * pruned / total,
            })
            total_pruned += pruned
            total_gates += total
        return {
            "per_layer": per_layer,
            "global_pruned": total_pruned,
            "global_total": total_gates,
            "global_sparsity_pct": 100.0 * total_pruned / total_gates,
            "threshold": threshold,
        }

    @torch.no_grad()
    def all_gate_values(self) -> torch.Tensor:
        """Flattened tensor of every gate value — used for histogram plots."""
        chunks = [torch.sigmoid(layer.gate_scores).flatten()
                  for layer in self.prunable_layers()]
        return torch.cat(chunks).cpu()
