import torch
from collections import defaultdict

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    # Expand advantages to match token shape
    advantages = advantages.expand_as(policy_log_probs)

    # Importance sampling ratio
    importance_ratio = torch.exp(policy_log_probs - old_log_probs)

    # Clipped ratio
    clipped_ratio = torch.clamp(
        importance_ratio,
        1 - cliprange,
        1 + cliprange
    )

    # Two objectives
    raw = importance_ratio * advantages
    clipped = clipped_ratio * advantages

    # Take minimum (PPO-style clipping)
    loss = -torch.min(raw, clipped)

    # Metadata: whether clipping was applied
    clipped_mask = clipped < raw

    metadata = {
        "clipped": clipped_mask
    }

    return loss, metadata