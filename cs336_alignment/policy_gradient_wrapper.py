from typing import Literal
import torch
from cs336_alignment.grpo_clip_loss import compute_grpo_clip_loss
from cs336_alignment.naive_policy_gradient import compute_naive_policy_gradient_loss

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards required for no_baseline"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages required for reinforce_with_baseline"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}

    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs required for grpo_clip"
        assert cliprange is not None, "cliprange required for grpo_clip"

        loss, metadata = compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
        )

    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return loss, metadata