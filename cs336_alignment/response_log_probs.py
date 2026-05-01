import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from cs336_alignment.compute_entropy import compute_entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:

    logits = model(input_ids).logits  # (B, T, V)

    log_probs = F.log_softmax(logits, dim=-1)

    gathered_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    # mask out padding tokens (here assumed to be -100)
    gathered_log_probs = gathered_log_probs.masked_fill(labels == -100, 0.0)

    result = {"log_probs": gathered_log_probs}

    if return_token_entropy:
        entropy = compute_entropy(logits)
        entropy = entropy.masked_fill(labels == 0, 0.0)
        result["token_entropy"] = entropy

    return result