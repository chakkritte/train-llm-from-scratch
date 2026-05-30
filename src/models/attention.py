import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.rope import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match the number of query heads for GQA.

    Args:
        x (torch.Tensor): Tensor of shape (batch, seq_len, n_kv_heads, head_dim).
        n_rep (int): Number of times to repeat each KV head.

    Returns:
        torch.Tensor: Tensor of shape (batch, seq_len, n_kv_heads * n_rep, head_dim).
    """
    batch, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Grouped Query Attention (GQA) with Rotary Position Embedding (RoPE).

    Uses PyTorch's native scaled_dot_product_attention (FlashAttention on
    supported hardware) for O(N) memory instead of materializing the full
    attention scores matrix.

    Args:
        n_embed (int): Embedding dimension.
        n_head (int): Number of query attention heads.
        n_kv_head (int): Number of key/value heads (can be < n_head for GQA).
        context_length (int): Maximum sequence length (unused, kept for API compat).
    """
    def __init__(self, n_embed: int, n_head: int, n_kv_head: int, context_length: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embed // n_head
        self.n_rep = n_head // n_kv_head

        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        assert n_embed % n_head == 0, "n_embed must be divisible by n_head"

        # Projections (no bias)
        self.wq = nn.Linear(n_embed, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(n_embed, n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(n_embed, n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, n_embed, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward pass through GQA attention.

        Args:
            x (torch.Tensor): Input of shape (batch, seq_len, n_embed).
            freqs_cis (torch.Tensor): RoPE frequencies of shape (seq_len, head_dim // 2).
            past_key_value (tuple, optional): Cached (key, value) from previous steps.

        Returns:
            tuple: (output tensor, updated past_key_value).
        """
        bsz, seqlen, _ = x.shape

        # Project to Q, K, V
        xq = self.wq(x).view(bsz, seqlen, self.n_head, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Concatenate with past KV if provided (for KV-cache generation)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)

        updated_kv = (xk, xv)

        # Repeat KV heads to match Q heads count for GQA
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Transpose for SDPA: (batch, n_head, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Memory-efficient attention.
        # is_causal=True only when Q and K have the same seq length (training / prompt pass).
        # When generating with KV-cache, seq lengths differ so we disable causal mask
        # (the query is already at the last position, so no future tokens exist).
        is_causal = (xq.size(-2) == xk.size(-2) and seqlen > 1)
        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=0.0, is_causal=is_causal)

        # Reshape back and project
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_head * self.head_dim)
        return self.wo(out), updated_kv


if __name__ == '__main__':
    batch = 2
    seq_len = 8
    n_embed = 512
    n_head = 8
    n_kv_head = 2
    head_dim = n_embed // n_head

    x = torch.randn(batch, seq_len, n_embed)
    freqs_cis = torch.polar(
        torch.ones(seq_len, head_dim // 2),
        torch.randn(seq_len, head_dim // 2),
    )

    attn = Attention(n_embed, n_head, n_kv_head, context_length=16)
    out, kv = attn(x, freqs_cis)
    print("Attention input shape:", x.shape)
    print("Attention output shape:", out.shape)
    print("KV cache K shape:", kv[0].shape, "V shape:", kv[1].shape)

    # Test KV-cache continuation
    x2 = torch.randn(batch, 1, n_embed)
    freqs_cis2 = torch.polar(torch.ones(1, head_dim // 2), torch.randn(1, head_dim // 2))
    out2, kv2 = attn(x2, freqs_cis2, past_key_value=kv)
    print("Cached generation output shape:", out2.shape)
