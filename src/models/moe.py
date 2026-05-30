import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MoeLayer(nn.Module):
    """
    Mixture of Experts (MoE) layer with top-k routing.

    Only k out of num_experts are active per token, keeping compute cost low
    while increasing total parameter count (and thus model capacity).

    Args:
        dim (int): Model dimension.
        num_experts (int): Total number of experts.
        top_k (int): Number of experts to use per token.
        hidden_dim (int): Hidden dimension of each expert FFN.
    """

    def __init__(self, dim: int, num_experts: int, top_k: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        # Router: projects tokens to expert scores
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Experts: each is a simple 2-layer FFN
        # Using nn.Parameter instead of nn.Linear for weight tying and speed
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden_dim, dim))

        # Initialize weights
        nn.init.kaiming_uniform_(self.w1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.w2, a=5 ** 0.5)

        # For load balancing loss
        self.register_buffer("_dummy", torch.tensor(0.0), persistent=False)

    def _expert_forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Forward through a single expert."""
        # x: (num_tokens, dim)
        h = torch.matmul(x, self.w1[expert_idx])
        h = F.gelu(h)
        h = torch.matmul(h, self.w2[expert_idx])
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            x (torch.Tensor): Input of shape (batch, seq_len, dim).

        Returns:
            Tuple of (output tensor, load_balance_loss).
        """
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (batch * seq_len, dim)

        # Compute routing scores
        router_logits = self.gate(x_flat)  # (num_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize

        # Accumulate outputs
        output = torch.zeros_like(x_flat)
        expert_token_counts = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens route to this expert
            token_mask = (top_k_indices == expert_idx).any(dim=-1)  # (num_tokens,)
            if not token_mask.any():
                continue

            expert_tokens = x_flat[token_mask]  # (num_tokens_for_expert, dim)
            expert_output = self._expert_forward(expert_tokens, expert_idx)

            # Compute weighted sum
            # Get the weight for this expert for each token
            token_positions = token_mask.nonzero(as_tuple=True)[0]
            for i, pos in enumerate(token_positions):
                # Find which position in top_k this expert is
                k_pos = (top_k_indices[pos] == expert_idx).nonzero(as_tuple=True)[0]
                if len(k_pos) > 0:
                    weight = top_k_probs[pos, k_pos[0]]
                    output[pos] += weight * expert_output[i]

            expert_token_counts[expert_idx] = token_mask.sum().float()

        # Compute load balancing loss
        # fraction of tokens routed to each expert
        avg_fraction = expert_token_counts / (bsz * seq_len + 1e-10)
        # average routing probability to each expert
        avg_prob = router_probs.mean(dim=0)
        # Load balance loss: encourages uniform distribution
        # Coefficient scales the loss
        aux_loss = self.num_experts * (avg_fraction * avg_prob).sum()

        return output.view(bsz, seq_len, dim), aux_loss


if __name__ == '__main__':
    batch, seq, dim = 2, 8, 512
    num_experts, top_k, hidden = 8, 2, 1408

    moe = MoeLayer(dim, num_experts, top_k, hidden)
    x = torch.randn(batch, seq, dim)
    out, aux_loss = moe(x)
    print(f"MoE input:  {x.shape}")
    print(f"MoE output: {out.shape}")
    print(f"Aux loss:   {aux_loss.item():.4f}")
    # Verify active params
    active_params = dim * hidden * 2 * top_k  # 2 matrices per expert * top_k experts
    total_params = dim * hidden * 2 * num_experts
    print(f"Active params/token: {active_params:,}")
    print(f"Total params: {total_params:,}")
