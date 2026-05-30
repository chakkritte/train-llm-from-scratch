import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Used in modern LLMs (Gemma, Llama, Qwen, Mistral) instead of standard LayerNorm.
    Normalizes by the RMS of the input and applies a learnable scale parameter.
    Does NOT subtract the mean (unlike LayerNorm).

    Args:
        dim (int): Dimensionality of the input to normalize.
        eps (float): Small constant for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Normalized and scaled tensor of the same shape.
        """
        # Compute RMS along the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm

if __name__ == '__main__':
    dim = 32
    rmsnorm = RMSNorm(dim)
    x = torch.randn(2, 5, dim)
    out = rmsnorm(x)
    print("RMSNorm input shape:", x.shape)
    print("RMSNorm output shape:", out.shape)
    # Verify that the mean squared is approximately 1
    print("Mean squared of normalized (should be ~1):", torch.mean(out ** 2).item())
