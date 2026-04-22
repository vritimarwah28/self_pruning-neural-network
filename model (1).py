%%writefile /content/model.py
"""
================================================================================
model.py  —  Self-Pruning Neural Network (CNN + Prunable FC Head)
================================================================================
Part of: Self-Pruning Neural Network on CIFAR-10
Project: Tredence Analytics — AI Engineer Case Study

Overview
--------
SelfPruningNet combines a standard CNN feature extractor with a prunable
fully-connected head. Only the FC layers use PrunableLinear — they learn
WHICH extracted features matter and prune the rest toward gate = 0.

Architecture
------------
[Conv Block 1]  Conv(3,32,3)  -> BN -> ReLU
                Conv(32,32,3) -> BN -> ReLU -> MaxPool(2) -> Dropout2d(0.2)

[Conv Block 2]  Conv(32,64,3) -> BN -> ReLU
                Conv(64,64,3) -> BN -> ReLU -> MaxPool(2) -> Dropout2d(0.2)

Flatten  (64 * 8 * 8 = 4096 features)

[Prunable FC Head]
  PrunableLinear(4096, 256) -> BN -> ReLU -> Dropout(0.3)
  PrunableLinear( 256, 128) -> BN -> ReLU -> Dropout(0.3)
  PrunableLinear( 128,  10)   [raw logits for 10 CIFAR-10 classes]

Why CNN + Prunable FC?
----------------------
A flat MLP on raw 3072 CIFAR-10 pixels tops out at ~45% accuracy — too low
to demonstrate a meaningful sparsity-accuracy trade-off. The CNN blocks
bring accuracy to ~65%, giving the prunable head enough signal to show a
clear trade-off as lambda increases.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from prunable_layer import PrunableLinear


class SelfPruningNet(nn.Module):
    """
    CNN feature extractor + prunable MLP classifier for CIFAR-10.

    The convolutional layers are standard (not prunable) — they extract
    spatial features efficiently.  The FC head uses PrunableLinear layers
    that learn to prune redundant connections via L1 gate regularisation.

    Parameters
    ----------
    None — architecture is fixed for CIFAR-10 (3 x 32 x 32 -> 10 classes)
    """

    def __init__(self):
        super().__init__()

        # ── Conv Block 1: 3 -> 32 channels, 32x32 -> 16x16 ───────────
        self.conv1a = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1a   = nn.BatchNorm2d(32)
        self.bn1b   = nn.BatchNorm2d(32)
        self.pool1  = nn.MaxPool2d(2, 2)      # 32x32 -> 16x16
        self.drop1  = nn.Dropout2d(0.2)

        # ── Conv Block 2: 32 -> 64 channels, 16x16 -> 8x8 ────────────
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2a   = nn.BatchNorm2d(64)
        self.bn2b   = nn.BatchNorm2d(64)
        self.pool2  = nn.MaxPool2d(2, 2)      # 16x16 -> 8x8
        self.drop2  = nn.Dropout2d(0.2)

        # After two MaxPool(2,2): spatial size = 8x8, channels = 64
        # Flattened dimension = 64 * 8 * 8 = 4096
        self.flatten = nn.Flatten()

        # ── Prunable FC Head ──────────────────────────────────────────
        self.fc1 = PrunableLinear(4096, 256)
        self.fc2 = PrunableLinear(256,  128)
        self.fc3 = PrunableLinear(128,   10)   # 10 output classes

        self.bn_fc1  = nn.BatchNorm1d(256)
        self.bn_fc2  = nn.BatchNorm1d(128)
        self.drop_fc = nn.Dropout(0.3)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Conv Block 1 ──────────────────────────────────────────────
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # ── Conv Block 2 ──────────────────────────────────────────────
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # ── Flatten: (B, 64, 8, 8) -> (B, 4096) ──────────────────────
        x = self.flatten(x)

        # ── Prunable FC Head ──────────────────────────────────────────
        x = self.drop_fc(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)     # raw logits; softmax applied inside cross-entropy
        return x

    # ------------------------------------------------------------------
    def prunable_layers(self):
        """Yield every PrunableLinear sub-module in the network."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def network_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Overall network sparsity = fraction of all gates below threshold.
        Only counts PrunableLinear layers (conv layers are not gated).
        """
        all_gates = torch.cat(
            [l.get_gates().flatten() for l in self.prunable_layers()]
        )
        return (all_gates < threshold).float().mean().item()
