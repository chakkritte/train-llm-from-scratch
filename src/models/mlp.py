import torch
import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):
    """
    SwiGLU feed-forward network used in modern LLMs (Gemma, Llama, Palmyra, Mixtral).

    Replaces the older ReLU-based MLP with a gated activation:
        hidden = SiLU(gate_proj(x)) * up_proj(x)
        out = down_proj(hidden)

    Args:
        n_embed (int): Input/output embedding dimension.
        intermediate_size (int): Hidden dimension of the FFN (typically ~2.7× n_embed).
    """
    def __init__(self, n_embed: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(n_embed, intermediate_size, bias=False)
        self.up_proj = nn.Linear(n_embed, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, n_embed, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through SwiGLU MLP.

        Args:
            x (torch.Tensor): Input of shape (B, T, n_embed).

        Returns:
            torch.Tensor: Output of shape (B, T, n_embed).
        """
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)


if __name__ == '__main__':
    batch = 2
    seq_len = 5
    n_embed = 512
    intermediate = 1408  # ~2.75×

    x = torch.randn(batch, seq_len, n_embed)
    mlp = MLP(n_embed, intermediate)
    out = mlp(x)
    print("SwiGLU MLP input shape:", x.shape)
    print("SwiGLU MLP output shape:", out.shape)
