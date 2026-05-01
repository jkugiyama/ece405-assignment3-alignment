import torch
from cs336_alignment.masked_normalize import masked_normalize

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # Per-sequence masked loss
    normalized_loss = masked_normalize(
        -policy_log_probs,
        response_mask,
        normalize_constant,
        dim=-1,  # sum over sequence
    )

    # Average over batch
    normalized_loss = normalized_loss.mean()

    # Scale for grad accumulation
    loss = normalized_loss / gradient_accumulation_steps

    loss.backward()

    return loss, {"normalized_loss": normalized_loss}