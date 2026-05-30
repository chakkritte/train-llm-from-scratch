import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (fast fused version).

    Uses rsqrt (fused) instead of sqrt + division for speed.
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RMSNorm using rsqrt for speed.
        """
        # rsqrt = 1/sqrt(x) is fused and faster than sqrt + divide
        x_norm = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x_norm

if __name__ == '__main__':
    dim = 32
    rmsnorm = RMSNorm(dim)
    x = torch.randn(2, 5, dim)
    out = rmsnorm(x)
    print("RMSNorm input shape:", x.shape)
    print("RMSNorm output shape:", out.shape)
    print("Mean squared of normalized (should be ~1):", torch.mean(out ** 2).item())
