import torch
import torch.nn as nn
from src.models.attention import Attention
from src.models.mlp import MLP
from src.models.rmsnorm import RMSNorm


class Block(nn.Module):
    """
    A single modern Transformer block with pre-RMSNorm, GQA, and SwiGLU or MoE.

    Architecture (Gemma / Llama 3 style):
        x = x + Attention(RMSNorm1(x))
        x = x + MLP/MoE(RMSNorm2(x))

    Args:
        n_embed (int): Embedding dimension.
        n_head (int): Number of query attention heads.
        n_kv_head (int): Number of key/value heads.
        intermediate_size (int): FFN hidden dimension.
        context_length (int): Maximum sequence length.
        moe (nn.Module, optional): Optional MoE layer replacing MLP.
    """
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_kv_head: int,
        intermediate_size: int,
        context_length: int,
        moe: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(n_embed)
        self.attn = Attention(n_embed, n_head, n_kv_head, context_length)
        self.ffn_norm = RMSNorm(n_embed)
        self.moe = moe
        self.mlp = None if moe is not None else MLP(n_embed, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
        """
        Forward pass through the block.

        Returns:
            tuple: (output tensor, updated KV cache, aux_loss).
        """
        attn_out, kv = self.attn(self.attn_norm(x), freqs_cis, past_key_value)
        x = x + attn_out

        if self.moe is not None:
            ffn_out, aux_loss = self.moe(self.ffn_norm(x))
            x = x + ffn_out
        else:
            x = x + self.mlp(self.ffn_norm(x))
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return x, kv, aux_loss


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
    # Real-valued freqs_cis for new rotate_half RoPE
    freqs_cis = torch.randn(seq_len, head_dim)

    # Standard block
    block = Block(n_embed, n_head, n_kv_head, intermediate, context_len)
    out, kv, aux = block(x, freqs_cis)
    print("Standard block output shape:", out.shape, "aux loss:", aux.item())

    # MoE block
    from src.models.moe import MoeLayer
    moe_layer = MoeLayer(n_embed, num_experts=8, top_k=2, hidden_dim=intermediate)
    moe_block = Block(n_embed, n_head, n_kv_head, intermediate, context_len, moe=moe_layer)
    out2, kv2, aux2 = moe_block(x, freqs_cis)
    print("MoE block output shape:", out2.shape, "aux loss:", aux2.item())
