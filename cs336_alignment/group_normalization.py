import torch
from collections import defaultdict

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    metadata = {}
    raw_rewards = []
    advantages = []

    rollout_batch_size = len(rollout_responses)

    for i in range(0, rollout_batch_size, group_size):
        group_responses = rollout_responses[i : i + group_size]
        group_ground_truths = repeated_ground_truths[i : i + group_size]

        group_rewards = []
        for response, ground_truth in zip(group_responses, group_ground_truths):
            reward_dict = reward_fn(response, ground_truth)
            r = reward_dict["reward"]

            group_rewards.append(r)
            raw_rewards.append(r)

        # convert to tensor for stable math
        group_rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)

        avg_reward = group_rewards_tensor.mean()
        std_reward = group_rewards_tensor.std(unbiased=True)  # important!

        for r in group_rewards_tensor:
            advantage = r - avg_reward
            if normalize_by_std:
                advantage = advantage / (std_reward + advantage_eps)
            advantages.append(advantage)

    # convert outputs to tensors
    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
    advantages_tensor = torch.stack(advantages)

    # better metadata (global stats)
    metadata = {
        "reward_mean": raw_rewards_tensor.mean().item(),
        "reward_std": raw_rewards_tensor.std(unbiased=False).item(),
        "reward_min": raw_rewards_tensor.min().item(),
        "reward_max": raw_rewards_tensor.max().item(),
    }

    return advantages_tensor, raw_rewards_tensor, metadata