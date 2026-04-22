%%writefile /content/prunable_layer.py
"""
================================================================================
prunable_layer.py  —  Custom PrunableLinear Layer with Gate Mechanism
================================================================================
Part of: Self-Pruning Neural Network on CIFAR-10
Project: Tredence Analytics — AI Engineer Case Study

Overview
--------
Implements PrunableLinear, a drop-in replacement for nn.Linear where every
weight w_ij is multiplied by a learnable gate g_ij = sigmoid(gate_score_ij).

    pruned_weight_ij = w_ij  *  sigmoid(s_ij)
    output           = x @ pruned_weight.T + bias

A gate value near 0  ->  weight is effectively pruned (removed).
A gate value near 1  ->  weight is kept unchanged.

Both `weight` and `gate_scores` are registered nn.Parameters so the
optimiser (Adam) updates them jointly and gradients flow through both.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with learnable gate parameters.

    Each weight w_ij is modulated by a gate g_ij = sigmoid(gate_score_ij):
        pruned_weight_ij = w_ij * sigmoid(s_ij)
        output           = x @ pruned_weight.T + bias

    Both weight and gate_scores are nn.Parameters — the optimiser updates
    them jointly and gradients flow through both paths.

    Key design choices
    ------------------
    * gate_scores initialised to +2  ->  sigmoid(2) ≈ 0.88
      Gates start nearly open so the network begins fully connected.
      The L1 sparsity loss then gradually drives redundant gates to 0.
      (Initialising to 0 gives sigmoid(0)=0.5 which already halves every
       weight before training even begins — too aggressive.)

    * No hard threshold during training — gates stay continuous so
      gradients always flow. Threshold is only used for *reporting* sparsity.

    Parameters
    ----------
    in_features  : int  - dimensionality of input
    out_features : int  - dimensionality of output
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias (same shapes as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores: same shape as weight.
        # Initialised to +2 so sigmoid(2) ≈ 0.88 — gates start nearly open.
        # The sparsity loss will push redundant gates toward -inf (gate -> 0).
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), 2.0)
        )

        # Kaiming uniform init for weights (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.

        Step 1: Squash raw gate_scores into (0,1) via Sigmoid
                gates = sigmoid(gate_scores)               shape: (out, in)

        Step 2: Element-wise gate masking
                pruned_weights = weight * gates            shape: (out, in)
                  - Gradient w.r.t. weight:      dL/dw = dL/d(pw) * gate
                  - Gradient w.r.t. gate_scores: dL/ds = dL/d(pw) * w * sigmoid'(s)
                Both paths are differentiable — optimizer updates both.

        Step 3: Standard linear transform with pruned weights
                output = x @ pruned_weights.T + bias
        """
        # Step 1
        gates = torch.sigmoid(self.gate_scores)      # (out_features, in_features)

        # Step 2
        pruned_weights = self.weight * gates          # (out_features, in_features)

        # Step 3
        return F.linear(x, pruned_weights, self.bias)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Return current gate values detached from computation graph."""
        return torch.sigmoid(self.gate_scores)

    @torch.no_grad()
    def layer_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Fraction of this layer's gates below `threshold`.
        A gate < 0.01 contributes < 1% of its weight — effectively pruned.
        """
        return (self.get_gates() < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"
