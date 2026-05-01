import random

import wandb

from cs336_alignment.evaluate_math_baseline import evaluate_vllm


def load_policy_into_vllm_instance(model, vllm_instance):
    """Copy current policy model weights into the vLLM inference instance."""
    state_dict = model.state_dict()
    llm_model = (
        vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
    )
    llm_model.load_weights(state_dict.items())


def log_generations(model, vllm, eval_prompts, eval_answers, reward_fn,
                    sampling_params, total_train_steps: int, num_samples: int = 100,
                    full_dataset: bool = False):

    load_policy_into_vllm_instance(model, vllm)

    # sample data
    if full_dataset:
        eval_prompts_sample = eval_prompts
        eval_answers_sample = eval_answers
    else:
        num_samples = min(num_samples, len(eval_prompts))
        indices = random.sample(range(len(eval_prompts)), num_samples)
        eval_prompts_sample = [eval_prompts[i] for i in indices]
        eval_answers_sample = [eval_answers[i] for i in indices]

    eval_results = evaluate_vllm(
        vllm, reward_fn, eval_prompts_sample, eval_answers_sample, sampling_params
    )

    correct_lengths = []
    incorrect_lengths = []
    format_reward = 0
    answer_reward = 0

    for info_dict in eval_results:
        if info_dict["answer_reward"] == 1:
            correct_lengths.append(len(info_dict["response"]))
        else:
            incorrect_lengths.append(len(info_dict["response"]))

        format_reward += info_dict["format_reward"]
        answer_reward += info_dict["answer_reward"]

    # length stats
    correct_length = sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0
    incorrect_length = sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0

    all_lengths = correct_lengths + incorrect_lengths
    average_length = sum(all_lengths) / len(all_lengths) if all_lengths else 0

    # reward stats
    num_results = len(eval_results)
    avg_format_reward = format_reward / num_results if num_results > 0 else 0
    avg_answer_reward = answer_reward / num_results if num_results > 0 else 0

    # logging
    wandb.log({
        "eval/correct_length": correct_length,
        "eval/incorrect_length": incorrect_length,
        "eval/average_length": average_length,
        "eval/format_reward": avg_format_reward,
        "eval/answer_reward": avg_answer_reward,
        "eval_step": total_train_steps
    })

    # print example
    if num_results > 0:
        sample = eval_results[0]
        print(f"Prompt: {sample['prompt']}")
        print(f"Answer: {sample['answer']}")
        print(f"Response: {sample['response']}")
        print(f"Format Reward: {sample['format_reward']}")
        print(f"Answer Reward: {sample['answer_reward']}")

    return avg_answer_reward, avg_format_reward