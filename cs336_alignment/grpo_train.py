import argparse
import importlib
import json
import logging
import random
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.group_normalization import compute_group_normalized_rewards
from cs336_alignment.grpo_microbatch_train import grpo_microbatch_train_step
from cs336_alignment.masked_mean import masked_mean
from cs336_alignment.response_log_probs import get_response_log_probs
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output

logger = logging.getLogger(__name__)


# Utilities
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_prompt_template(prompt_template_path: str) -> str:
    candidate = Path(prompt_template_path)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    repo_candidate = repo_root / prompt_template_path
    if repo_candidate.exists():
        return repo_candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")


def load_math_split(path: str) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    out: list[dict[str, str]] = []
    for row in rows:
        problem = row.get("problem") or row.get("question")
        answer = row.get("answer") or row.get("solution")
        if problem and answer:
            out.append({"question": str(problem), "answer": str(answer)})
    if not out:
        raise ValueError(f"No usable examples in {path}")
    return out


def truncate_at_second_answer_tag(text: str) -> str:
    tag = "</answer>"
    idx = text.find(tag)
    if idx == -1:
        return text
    idx2 = text.find(tag, idx + len(tag))
    if idx2 == -1:
        return text
    return text[: idx2 + len(tag)]


# vLLM helpers
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm = importlib.import_module("vllm")
    me = importlib.import_module("vllm.model_executor")
    LLM = getattr(vllm, "LLM")
    getattr(me, "set_random_seed")(seed)
    ws_patch = patch("torch.distributed.get_world_size", return_value=1)
    prof_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with ws_patch, prof_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.float32,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm(policy, llm) -> None:
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(policy.state_dict().items())


def run_vllm(llm, prompts: list[str], sampling_params) -> list[str]:
    outputs = llm.generate(prompts, sampling_params)
    return [out.text for resp in outputs for out in resp.outputs]


# Tokenisation helper for a rollout batch

def tokenize_batch(prompts: list[str], responses: list[str], tokenizer, max_length: int, device: torch.device):
    tokenized = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer,
    )
    return {
        "input_ids": tokenized["input_ids"][:, :max_length].to(device),
        "labels": tokenized["labels"][:, :max_length].to(device),
        "response_mask": tokenized["response_mask"][:, :max_length].to(device),
    }


# Validation evaluation
@torch.no_grad()
def evaluate(
    policy,
    llm,
    template: str,
    valid_data: list[dict[str, str]],
    max_eval_examples: int,
    max_new_tokens: int,
) -> dict[str, float]:
    vllm = importlib.import_module("vllm")
    SamplingParams = getattr(vllm, "SamplingParams")

    subset = valid_data[:max_eval_examples]
    prompts = [template.format(question=x["question"]) for x in subset]
    answers = [x["answer"] for x in subset]

    load_policy_into_vllm(policy, llm)
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    responses = [truncate_at_second_answer_tag(r) for r in run_vllm(llm, prompts, params)]

    answer_rewards, format_rewards = [], []
    for response, answer in zip(responses, answers):
        m = r1_zero_reward_fn(response, answer)
        answer_rewards.append(float(m["answer_reward"]))
        format_rewards.append(float(m["format_reward"]))

    return {
        "answer_accuracy": sum(answer_rewards) / len(answer_rewards),
        "format_accuracy": sum(format_rewards) / len(format_rewards),
        "reward_mean": sum(answer_rewards) / len(answer_rewards),
    }, responses[:5], prompts[:5]


# Main training loop
def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_math_split(args.train_jsonl_path)
    valid_data = load_math_split(args.validation_jsonl_path)
    template = resolve_prompt_template(args.prompt_template_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_device = torch.device(args.policy_device)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    policy.to(policy_device)

    optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    rollout_batch_size = args.rollout_batch_size
    if rollout_batch_size is None:
        rollout_batch_size = args.questions_per_step * args.rollout_g
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be positive")
    if rollout_batch_size % args.rollout_g != 0:
        raise ValueError("rollout_batch_size must be divisible by rollout_g")

    train_batch_size = args.train_batch_size if args.train_batch_size is not None else args.per_device_batch_size
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    if args.epochs_per_rollout_batch <= 0:
        raise ValueError("epochs_per_rollout_batch must be positive")

    llm = init_vllm(args.model_name_or_path, args.vllm_device, args.seed, args.vllm_gpu_memory_utilization)

    vllm = importlib.import_module("vllm")
    SamplingParams = getattr(vllm, "SamplingParams")
    rollout_params = SamplingParams(
        temperature=args.rollout_temperature,
        top_p=1.0,
        max_tokens=args.rollout_max_new_tokens,
        n=args.rollout_g,
    )

    reward_curve: list[dict[str, Any]] = []
    val_curve: list[dict[str, Any]] = []
    example_rollouts: list[dict[str, Any]] = []

    global_step = 0
    pbar = tqdm(total=args.max_train_steps, desc="GRPO training")

    while global_step < args.max_train_steps:
        policy.train()

        # 1. Sample prompts so the flattened rollout has size rollout_batch_size = n_prompts * rollout_g
        n_prompts_per_rollout_batch = rollout_batch_size // args.rollout_g
        batch = random.sample(train_data, k=min(n_prompts_per_rollout_batch, len(train_data)))
        prompts = [template.format(question=x["question"]) for x in batch]
        answers = [x["answer"] for x in batch]

        # 2. Rollout G responses per question via vLLM
        load_policy_into_vllm(policy, llm)
        all_responses_raw = run_vllm(llm, prompts, rollout_params)

        # Reshape: (Q*G,) -> handled flat
        flat_prompts = [p for p in prompts for _ in range(args.rollout_g)]
        flat_answers = [a for a in answers for _ in range(args.rollout_g)]
        flat_responses = [truncate_at_second_answer_tag(r) for r in all_responses_raw]

        # 3. Compute group-normalised advantages
        advantages_tensor, raw_rewards_tensor, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=flat_responses,
            repeated_ground_truths=flat_answers,
            group_size=args.rollout_g,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.normalize_by_std,
        )

        reward_curve.append({
            "step": global_step,
            "reward_mean": reward_meta["reward_mean"],
            "reward_std": reward_meta["reward_std"],
        })
        logger.info("step=%d reward_mean=%.4f reward_std=%.4f", global_step, reward_meta["reward_mean"], reward_meta["reward_std"])

        # 4. Tokenise the full rollout batch
        tok_batch = tokenize_batch(flat_prompts, flat_responses, tokenizer, args.max_length, policy_device)
        input_ids = tok_batch["input_ids"]
        labels = tok_batch["labels"]
        response_mask = tok_batch["response_mask"]

        # 5. Old log-probs from the current policy before any optimizer updates (off-policy baseline)
        policy.eval()
        with torch.inference_mode():
            old_out = get_response_log_probs(model=policy, input_ids=input_ids, labels=labels)
            old_log_probs = old_out["log_probs"].detach()
        policy.train()

        # 6. Per-token advantages: broadcast (B,) -> (B, T) using response_mask
        adv = advantages_tensor.to(policy_device).unsqueeze(1).expand_as(response_mask).float()

        # 7. Off-policy inner loop: multiple epochs / minibatches over fixed rollout data
        n = input_ids.size(0)
        micro_bs = train_batch_size
        grad_norm_value = 0.0
        ent = None
        update_count = 0

        for epoch_idx in range(args.epochs_per_rollout_batch):
            shuffled_indices = list(range(n))
            random.shuffle(shuffled_indices)

            for mb_start in range(0, n, micro_bs):
                mb_indices = shuffled_indices[mb_start: mb_start + micro_bs]
                if not mb_indices:
                    continue

                optimizer.zero_grad(set_to_none=True)

                mb_input_ids = input_ids[mb_indices]
                mb_labels = labels[mb_indices]
                mb_mask = response_mask[mb_indices]
                mb_adv = adv[mb_indices]
                mb_old_lp = old_log_probs[mb_indices]

                policy_out = get_response_log_probs(
                    model=policy,
                    input_ids=mb_input_ids,
                    labels=mb_labels,
                    return_token_entropy=(ent is None),
                )

                grpo_microbatch_train_step(
                    policy_log_probs=policy_out["log_probs"],
                    response_mask=mb_mask,
                    gradient_accumulation_steps=1,
                    loss_type=args.loss_type,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    cliprange=args.cliprange,
                )

                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                grad_norm_value = float(grad_norm.detach().cpu())
                optimizer.step()
                update_count += 1

                if "token_entropy" in policy_out and ent is None:
                    ent = masked_mean(
                        policy_out["token_entropy"].detach(),
                        mb_mask.to(policy_device),
                    ).item()

        # 8. Log entropy from first micro-batch
        if ent is not None:
            reward_curve[-1]["response_entropy"] = ent
        reward_curve[-1]["grad_norm"] = grad_norm_value
        reward_curve[-1]["num_updates"] = update_count

        global_step += 1
        pbar.update(1)

        # 9. Periodic validation + example rollouts
        if global_step % args.eval_every_steps == 0 or global_step == args.max_train_steps:
            policy.eval()
            val_metrics, sample_responses, sample_prompts = evaluate(
                policy=policy,
                llm=llm,
                template=template,
                valid_data=valid_data,
                max_eval_examples=args.max_eval_examples,
                max_new_tokens=args.eval_max_new_tokens,
            )
            policy.train()
            val_curve.append({"step": global_step, **val_metrics})
            logger.info("eval step=%d answer_acc=%.4f", global_step, val_metrics["answer_accuracy"])

            example_rollouts.append({
                "step": global_step,
                "examples": [
                    {"prompt": p[:200], "response": r[:400]}
                    for p, r in zip(sample_prompts, sample_responses)
                ],
            })

    pbar.close()

    # Save outputs
    with open(out_dir / "train_reward_curve.json", "w") as f:
        json.dump(reward_curve, f, indent=2)
    with open(out_dir / "val_curve.json", "w") as f:
        json.dump(val_curve, f, indent=2)
    with open(out_dir / "example_rollouts.json", "w") as f:
        json.dump(example_rollouts, f, indent=2)

    policy.save_pretrained(out_dir / "policy")
    tokenizer.save_pretrained(out_dir / "policy")
    logger.info("Done. Outputs in %s", out_dir)


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training loop on MATH")
    p.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--train-jsonl-path", default="/data/a5-alignment/MATH/train.jsonl")
    p.add_argument("--validation-jsonl-path", default="/data/a5-alignment/MATH/validation.jsonl")
    p.add_argument("--prompt-template-path", default="cs336_alignment/prompts/r1_zero.prompt")
    p.add_argument("--output-dir", default="outputs/grpo_math")

    p.add_argument("--loss-type", default="grpo_clip",
                   choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    p.add_argument("--rollout-g", type=int, default=8)
    p.add_argument("--rollout-batch-size", type=int, default=None,
                   help="Total number of sampled responses per rollout batch (must be divisible by rollout-g).")
    p.add_argument("--epochs-per-rollout-batch", type=int, default=1,
                   help="Number of optimizer epochs over a fixed rollout batch.")
    p.add_argument("--train-batch-size", type=int, default=None,
                   help="Minibatch size for policy updates inside each rollout batch.")
    p.add_argument("--questions-per-step", type=int, default=16)
    p.add_argument("--rollout-temperature", type=float, default=1.0)
    p.add_argument("--rollout-max-new-tokens", type=int, default=1024)
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--advantage-eps", type=float, default=1e-8)
    p.add_argument("--normalize-by-std", action="store_true", default=True)

    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-train-steps", type=int, default=200)

    p.add_argument("--eval-every-steps", type=int, default=20)
    p.add_argument("--max-eval-examples", type=int, default=256)
    p.add_argument("--eval-max-new-tokens", type=int, default=1024)

    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--vllm-device", default="cuda:0")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    args = parse_args()
    train(args)
