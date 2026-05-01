import torch
from typing import Literal
from cs336_alignment.policy_gradient_wrapper import compute_policy_gradient_loss

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_unclipped"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    use_mask_normalize: bool | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # 1. Compute per-token loss
    policy_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )

    # 2. Mask invalid tokens
    masked_loss = policy_loss * response_mask

    # 3. Reduce properly over tokens
    # avoid divide-by-zero
    denom = response_mask.sum().clamp_min(1.0)
    response_policy_loss = masked_loss.sum() / denom

    # 4. Scale for gradient accumulation
    scaled_loss = response_policy_loss / gradient_accumulation_steps

    # 5. Backward pass
    scaled_loss.backward()

    # 6. Return detached scalar for logging
    return scaled_loss.detach(), metadata