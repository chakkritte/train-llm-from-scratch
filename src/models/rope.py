import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dimensions of the input.
    Real-valued alternative to complex RoPE — faster and numerically stable in FP16.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors using real-valued rotation.

    Args:
        xq (torch.Tensor): Query tensor of shape (batch, seq_len, n_head, head_dim).
        xk (torch.Tensor): Key tensor of shape (batch, seq_len, n_kv_head, head_dim).
        freqs_cis (torch.Tensor): Precomputed (cos, sin) pairs, shape (seq_len, head_dim * 2).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    head_dim = xq.shape[-1]
    cos = freqs_cis[..., : head_dim]  # (seq_len, head_dim)
    sin = freqs_cis[..., head_dim :]  # (seq_len, head_dim)

    # Broadcast cos/sin to match xq/xk shape
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)

    xq_rot = xq * cos + rotate_half(xq) * sin
    xk_rot = xk * cos + rotate_half(xk) * sin
    return xq_rot, xk_rot


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute (cos, sin) frequency tensor for real-valued RoPE.

    Returns tensor of shape (end, dim * 2) where first half = cos, second half = sin.
    cos/sin values are repeated to match head_dim (e.g., [c0, c0, c1, c1, ...]).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)  # (end, dim // 2)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (end, dim)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # (end, dim)
    return torch.cat([cos, sin], dim=-1)  # (end, dim * 2)


class RotaryEmbedding(nn.Module):
    """
    Wrapper module for real-valued Rotary Position Embedding.
    Precomputes and caches (cos, sin) up to max_seq_len.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return the precomputed freqs_cis for the given sequence length."""
        return self.freqs_cis[:seq_len]


if __name__ == '__main__':
    batch = 2
    seq_len = 8
    n_head = 4
    head_dim = 64

    xq = torch.randn(batch, seq_len, n_head, head_dim)
    xk = torch.randn(batch, seq_len, n_head, head_dim)

    rope = RotaryEmbedding(head_dim, max_seq_len=seq_len)
    freqs_cis = rope(seq_len)

    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
    print("RoPE Q shape:", xq_rot.shape)
    print("RoPE K shape:", xk_rot.shape)
    # RoPE should preserve norms
    print("Norm preserved (Q):", torch.allclose(xq.norm(), xq_rot.norm(), atol=1e-4))
    print("Norm preserved (K):", torch.allclose(xk.norm(), xk_rot.norm(), atol=1e-4))
