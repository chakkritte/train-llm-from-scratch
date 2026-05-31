"""
Fast inference engine inspired by custom-kernel inference stacks.

Adapts the "custom native ops + flat weights + autoregressive runtime" idea
to PyTorch by:

1. torch.compile()-ing the single-token forward pass so the compiler can fuse
   RMSNorm, projections, and activations into custom CUDA kernels.
2. Truncating KV-cache to a fixed max length to avoid dynamic-shape
   recompilations during generation.
3. Keeping the generation loop in minimal Python overhead.

This does NOT replace PyTorch — it uses PyTorch 2.x graph compilation as the
modern, Python-native equivalent of hand-writing fused CUDA kernels.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.transformer import Transformer


class FastInferenceEngine:
    """
    Optimized inference engine that compiles the hot forward path and
    truncates KV-cache to keep tensor shapes static.
    """

    def __init__(
        self,
        model: Transformer,
        max_seq_len: int,
        compile: bool = True,
        device: str = "cuda",
    ) -> None:
        self.model = model.eval().to(device)
        self.max_seq_len = max_seq_len
        self.device = device
        self.n_layers = model.n_layers

        # Compile the single-token forward (the hot path)
        self._forward_fn = model
        if compile and hasattr(torch, "compile"):
            # fullgraph=False because KV-cache concatenation and mutation are
            # not always expressible as a pure functional graph.
            self._forward_fn = torch.compile(
                model, mode="max-autotune", fullgraph=False
            )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with compiled single-token forward and
        static-shape KV-cache management.

        Args:
            idx: Initial token indices (B, T).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Optional top-k filtering.
            top_p: Optional nucleus (top-p) filtering.

        Returns:
            Extended token indices (B, T + max_new_tokens).
        """
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        for _ in range(max_new_tokens):
            # KV-cache: first pass uses full context, later passes use 1 token
            if past_kvs is None:
                idx_cond = idx[:, -self.max_seq_len :]
            else:
                idx_cond = idx[:, -1:]

            logits, _, past_kvs = self._forward_fn(
                idx_cond,
                past_key_values=past_kvs,
                use_cache=True,
            )

            # Truncate KV-cache to max_seq_len so shapes stay static
            # (avoids torch.compile recompilation as seq_len grows)
            if past_kvs is not None:
                past_kvs = [
                    (
                        k[:, -self.max_seq_len :],
                        v[:, -self.max_seq_len :],
                    )
                    for k, v in past_kvs
                ]

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
                    sorted_probs, sorted_indices = torch.sort(
                        probs, descending=True, dim=-1
                    )
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum > top_p
                    # Shift mask so we keep the first token that crosses threshold
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    sorted_probs[mask] = 0
                    probs.scatter_(-1, sorted_indices, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
