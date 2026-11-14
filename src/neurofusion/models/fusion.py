"""Feature fusion and channel recalibration on already-pooled vectors."""
from __future__ import annotations

import torch
from torch import nn


class SEGate(nn.Module):
    """Squeeze-and-excitation-style gate on a pooled feature vector (manuscript 3.3.3, Table 2).

        g = sigmoid(W2 relu(W1 z));  out = z * g

    z is already a pooled [B, D] vector, so no averaging across the feature dimension is applied.
    Reduction ratio r gives hidden size D // r (r=8 -> 144 for D=1152).
    """

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        if dim % reduction:
            raise ValueError(f"dim {dim} not divisible by reduction {reduction}")
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)

    def gate(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(z))))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.gate(z)


class MLPControl(nn.Module):
    """Capacity-comparable control WITHOUT multiplicative gating (new extension, not in the manuscript).

    Same two linear layers as :class:`SEGate`. ``residual``: out = z + fc2(relu(fc1(z))); ``replace``:
    out = fc2(relu(fc1(z))). The choice is a design decision of this repository.
    """

    def __init__(self, dim: int, reduction: int = 8, mode: str = "residual"):
        super().__init__()
        if mode not in {"residual", "replace"}:
            raise ValueError(mode)
        self.mode = mode
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc2(torch.relu(self.fc1(z)))
        return z + h if self.mode == "residual" else h


class ConcatFusion(nn.Module):
    """Concatenate along the feature dimension (768 + 384 = 1152)."""

    def __init__(self, dim_a: int, dim_b: int):
        super().__init__()
        self.out_dim = dim_a + dim_b

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=1)


class SumFusion(nn.Module):
    """Element-wise summation after projecting the smaller branch to the larger dimension.

    The manuscript names this ablation but not the alignment. Projecting DINOv2 (384) to the CoAtNet dimension
    (768) with one Linear layer reproduces the manuscript's reported parameter gap between the summation model
    (49.11 M) and the concatenation model (49.20 M) to within rounding; alternatives (e.g. projecting 768 -> 384)
    do not. That agreement is consistent with, but does not prove, the original design.
    """

    def __init__(self, dim_a: int, dim_b: int):
        super().__init__()
        self.out_dim = max(dim_a, dim_b)
        self.proj_a = nn.Identity() if dim_a == self.out_dim else nn.Linear(dim_a, self.out_dim)
        self.proj_b = nn.Identity() if dim_b == self.out_dim else nn.Linear(dim_b, self.out_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.proj_a(a) + self.proj_b(b)
