import torch

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    # Ensure rewards/advantages broadcast across sequence_length
    # Shape: (batch_size, 1) -> (batch_size, sequence_length)
    expanded_rewards = raw_rewards_or_advantages.expand_as(policy_log_probs)

    # Policy gradient loss: - advantage * log_prob
    loss = -expanded_rewards * policy_log_probs

    return loss