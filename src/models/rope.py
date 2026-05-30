import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This is the core of Rotary Position Embedding (RoPE). It computes a complex
    rotation matrix that is applied to query and key vectors to encode position.

    Args:
        dim (int): Dimensionality of the embedding to rotate (must be even).
        end (int): Maximum sequence length to precompute for.
        theta (float): Base for the frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequencies of shape (end, dim // 2, 2) as real pairs.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)  # (end, dim // 2)
    # Represent as real pairs instead of complex for AMP / FP16 compatibility
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (end, dim // 2) complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape freqs_cis to broadcast with x for rotary embedding application.

    Args:
        freqs_cis (torch.Tensor): Precomputed frequencies, shape (seq_len, dim // 2).
        x (torch.Tensor): Query or key tensor, shape (batch, seq_len, n_head, head_dim).

    Returns:
        torch.Tensor: Reshaped freqs_cis ready for broadcasting.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        xq (torch.Tensor): Query tensor of shape (batch, seq_len, n_head, head_dim).
        xk (torch.Tensor): Key tensor of shape (batch, seq_len, n_kv_head, head_dim).
        freqs_cis (torch.Tensor): Precomputed frequencies, shape (seq_len, head_dim // 2).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    # View as complex numbers: (..., head_dim//2, 2) -> complex view
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # Apply rotation via complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    """
    Wrapper module for Rotary Position Embedding.

    Precomputes and caches freq_cis up to max_seq_len.
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
