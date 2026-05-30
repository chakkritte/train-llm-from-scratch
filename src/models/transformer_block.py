import torch
import torch.nn as nn
from src.models.attention import Attention
from src.models.mlp import MLP
from src.models.rmsnorm import RMSNorm


class Block(nn.Module):
    """
    A single modern Transformer block with pre-RMSNorm, GQA, and SwiGLU.

    Architecture (Gemma / Llama 3 style):
        x = x + Attention(RMSNorm1(x))
        x = x + MLP(RMSNorm2(x))

    Args:
        n_embed (int): Embedding dimension.
        n_head (int): Number of query attention heads.
        n_kv_head (int): Number of key/value heads.
        intermediate_size (int): FFN hidden dimension.
        context_length (int): Maximum sequence length.
    """
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_kv_head: int,
        intermediate_size: int,
        context_length: int,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(n_embed)
        self.attn = Attention(n_embed, n_head, n_kv_head, context_length)
        self.mlp_norm = RMSNorm(n_embed)
        self.mlp = MLP(n_embed, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward pass through the block.

        Args:
            x (torch.Tensor): Input of shape (B, T, C).
            freqs_cis (torch.Tensor): RoPE frequencies for the current sequence length.
            past_key_value (tuple, optional): Cached KV from previous generation steps.

        Returns:
            tuple: (output tensor, updated KV cache).
        """
        attn_out, kv = self.attn(self.attn_norm(x), freqs_cis, past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, kv


if __name__ == '__main__':
    batch = 2
    seq_len = 8
    n_embed = 512
    n_head = 8
    n_kv_head = 2
    intermediate = 1408
    context_len = 16
    head_dim = n_embed // n_head

    x = torch.randn(batch, seq_len, n_embed)
    freqs_cis = torch.polar(
        torch.ones(seq_len, head_dim // 2),
        torch.randn(seq_len, head_dim // 2),
    )

    block = Block(n_embed, n_head, n_kv_head, intermediate, context_len)
    out, kv = block(x, freqs_cis)
    print("Block input shape:", x.shape)
    print("Block output shape:", out.shape)

    # Test KV-cache continuation
    x2 = torch.randn(batch, 1, n_embed)
    freqs_cis2 = torch.polar(torch.ones(1, head_dim // 2), torch.randn(1, head_dim // 2))
    out2, kv2 = block(x2, freqs_cis2, past_key_value=kv)
    print("Cached block output shape:", out2.shape)
