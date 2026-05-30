import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.transformer_block import Block
from src.models.rmsnorm import RMSNorm
from src.models.rope import RotaryEmbedding


class Transformer(nn.Module):
    """
    Modern decoder-only Transformer (Gemma / Llama 3 style).

    Features:
    - Rotary Position Embedding (RoPE) inside attention
    - RMSNorm instead of LayerNorm
    - Grouped Query Attention (GQA)
    - SwiGLU feed-forward network
    - Optional weight tying between token embeddings and LM head
    - KV-cache support for efficient autoregressive generation
    - Gradient checkpointing for memory-efficient training of large models

    Args:
        vocab_size (int): Vocabulary size.
        n_embed (int): Embedding dimension.
        n_layers (int): Number of transformer blocks.
        n_head (int): Number of query attention heads.
        n_kv_head (int): Number of key/value heads (<= n_head).
        intermediate_size (int): FFN hidden dimension.
        context_length (int): Maximum sequence length.
        tie_weights (bool): Whether to tie token_embed and lm_head weights.
        rope_theta (float): Base for RoPE frequency computation.
        use_gradient_checkpointing (bool): Enable gradient checkpointing to trade compute for memory.
    """
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_layers: int,
        n_head: int,
        n_kv_head: int,
        intermediate_size: int,
        context_length: int,
        tie_weights: bool = True,
        rope_theta: float = 10000.0,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_layers = n_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.rope = RotaryEmbedding(n_embed // n_head, max_seq_len=context_length, theta=rope_theta)

        self.blocks = nn.ModuleList(
            [
                Block(n_embed, n_head, n_kv_head, intermediate_size, context_length)
                for _ in range(n_layers)
            ]
        )

        self.norm = RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and embedding layers with small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Forward pass through the Transformer.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T).
            targets (torch.Tensor, optional): Target token indices for loss. Defaults to None.
            past_key_values (list, optional): Cached KV states from previous generation steps.
            use_cache (bool): Whether to return updated KV caches.

        Returns:
            tuple: (logits, loss or None, past_key_values or None)
        """
        bsz, seqlen = idx.shape
        h = self.token_embed(idx)

        # RoPE for the current sequence length
        freqs_cis = self.rope(seqlen)

        new_past_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if self.training and self.use_gradient_checkpointing and not use_cache:
                # Gradient checkpointing: recompute block forward during backward
                # to save activation memory. Not used with KV-cache (inference).
                h, kv = torch.utils.checkpoint.checkpoint(
                    block, h, freqs_cis, past_kv, use_reentrant=False
                )
            else:
                h, kv = block(h, freqs_cis, past_key_value=past_kv)
            if use_cache:
                new_past_kvs.append(kv)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1).long())
            return logits, loss, None

        if use_cache:
            return logits, None, new_past_kvs
        return logits, None, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively with KV-cache.

        Args:
            idx (torch.Tensor): Initial sequence of token indices (B, T).
            max_new_tokens (int): Number of new tokens to generate.
            temperature (float): Sampling temperature. 0 = greedy.
            top_k (int, optional): Top-k filtering.
            top_p (float, optional): Nucleus (top-p) filtering.

        Returns:
            torch.Tensor: Extended sequence of token indices.
        """
        past_kvs = None
        for _ in range(max_new_tokens):
            # KV-cache: first pass uses full context, later passes use only last token
            if past_kvs is None:
                idx_cond = idx[:, -self.context_length:]
                logits, _, past_kvs = self(idx_cond, use_cache=True)
            else:
                idx_cond = idx[:, -1:]
                logits, _, past_kvs = self(idx_cond, past_key_values=past_kvs, use_cache=True)

            logits = logits[:, -1, :]

            # Greedy decoding
            if temperature == 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)

                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs[probs < v[:, [-1]]] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                # Top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum > top_p
                    # Shift mask so we keep the first token that crosses the threshold
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    sorted_probs[mask] = 0
                    probs.scatter_(-1, sorted_indices, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    vocab_size = 100
    n_embed = 256
    n_layers = 4
    n_head = 8
    n_kv_head = 2
    intermediate = 704
    context_len = 16

    model = Transformer(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layers=n_layers,
        n_head=n_head,
        n_kv_head=n_kv_head,
        intermediate_size=intermediate,
        context_length=context_len,
        tie_weights=True,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}")

    # Training forward
    batch = 2
    seq_len = 8
    x = torch.randint(0, vocab_size, (batch, seq_len))
    y = torch.randint(0, vocab_size, (batch, seq_len))
    logits, loss, _ = model(x, targets=y)
    print("Training logits shape:", logits.shape)
    print("Training loss:", loss.item())

    # Generation with KV-cache
    start = x[:, :1]
    generated = model.generate(start, max_new_tokens=5, temperature=0.8, top_k=10)
    print("Generated shape:", generated.shape)
