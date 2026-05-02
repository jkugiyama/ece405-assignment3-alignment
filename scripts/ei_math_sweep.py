import argparse
import importlib
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.masked_mean import masked_mean
from cs336_alignment.response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train import sft_microbatch_train_step
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output

logger = logging.getLogger(__name__)


class PromptResponseDataset(Dataset):
    def __init__(self, examples: list[dict[str, str]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.examples[idx]


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
            if not line:
                continue
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
        if problem is None or answer is None:
            continue
        out.append({"question": str(problem), "answer": str(answer)})
    if not out:
        raise ValueError(f"No usable examples in {path}")
    return out


def truncate_at_nth_answer_tag(text: str, n: int = 2) -> str:
    tag = "</answer>"
    idx = -1
    start = 0
    for _ in range(n):
        idx = text.find(tag, start)
        if idx == -1:
            return text
        start = idx + len(tag)
    return text[: start]


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_module = importlib.import_module("vllm")
    model_executor_module = importlib.import_module("vllm.model_executor")
    LLM = getattr(vllm_module, "LLM")
    vllm_set_random_seed = getattr(model_executor_module, "set_random_seed")

    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def run_vllm(vllm_model, prompts: list[str], sampling_params) -> list[str]:
    outputs = vllm_model.generate(prompts, sampling_params)
    return [output.text for response in outputs for output in response.outputs]


def make_collate_fn(tokenizer, max_length: int):
    def collate(examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompt_strs = [x["prompt"] for x in examples]
        output_strs = [x["response"] for x in examples]

        tokenized = tokenize_prompt_and_output(
            prompt_strs=prompt_strs,
            output_strs=output_strs,
            tokenizer=tokenizer,
        )

        return {
            "input_ids": tokenized["input_ids"][:, :max_length],
            "labels": tokenized["labels"][:, :max_length],
            "response_mask": tokenized["response_mask"][:, :max_length],
        }

    return collate


def compute_entropy_on_dataset(policy, tokenizer, examples: list[dict[str, str]], max_length: int, device: torch.device) -> float:
    if not examples:
        return 0.0
    collate = make_collate_fn(tokenizer, max_length)
    batch = collate(examples)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)

    out = get_response_log_probs(model=policy, input_ids=input_ids, labels=labels, return_token_entropy=True)
    seq_entropy = masked_mean(out["token_entropy"], response_mask, dim=-1)
    return float(seq_entropy.mean().detach().cpu())


@torch.no_grad()
def evaluate_accuracy(policy, llm, template: str, valid_data: list[dict[str, str]], max_eval_examples: int, max_new_tokens: int) -> float:
    vllm_module = importlib.import_module("vllm")
    SamplingParams = getattr(vllm_module, "SamplingParams")

    subset = valid_data[:max_eval_examples] if max_eval_examples > 0 else valid_data
    prompts = [template.format(question=x["question"]) for x in subset]
    answers = [x["answer"] for x in subset]

    load_policy_into_vllm_instance(policy, llm)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    responses = run_vllm(llm, prompts, sampling_params)
    responses = [truncate_at_nth_answer_tag(x, n=2) for x in responses]

    correct = 0.0
    for response, answer in zip(responses, answers):
        correct += float(r1_zero_reward_fn(response, answer)["answer_reward"])
    return correct / max(1, len(answers))


def run_single_config(args: argparse.Namespace, rollout_g: int, sft_epochs: int, db_size: int, run_dir: Path) -> dict[str, Any]:
    set_seed(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

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
    policy.train()

    optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    llm = init_vllm(
        model_id=args.model_name_or_path,
        device=args.vllm_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    vllm_module = importlib.import_module("vllm")
    SamplingParams = getattr(vllm_module, "SamplingParams")

    rollout_sampling = SamplingParams(
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        max_tokens=args.rollout_max_new_tokens,
    )

    acc_curve: list[dict[str, float | int]] = []
    ent_curve: list[dict[str, float | int]] = []

    for ei_step in range(1, args.n_ei_steps + 1):
        batch = random.sample(train_data, k=min(db_size, len(train_data)))
        batch_prompts = [template.format(question=x["question"]) for x in batch]

        rollout_prompts: list[str] = []
        meta: list[tuple[int, str, str]] = []
        for idx, row in enumerate(batch):
            for _ in range(rollout_g):
                rollout_prompts.append(batch_prompts[idx])
                meta.append((idx, row["question"], row["answer"]))

        load_policy_into_vllm_instance(policy, llm)
        raw_rollouts = run_vllm(llm, rollout_prompts, rollout_sampling)
        rollouts = [truncate_at_nth_answer_tag(x, n=2) for x in raw_rollouts]

        by_question: dict[int, list[dict[str, Any]]] = {}
        for m, response in zip(meta, rollouts):
            qidx, question, answer = m
            score = float(r1_zero_reward_fn(response, answer)["reward"])
            by_question.setdefault(qidx, []).append(
                {
                    "prompt": template.format(question=question),
                    "response": response,
                    "score": score,
                }
            )

        expert_pairs: list[dict[str, str]] = []
        for qidx in sorted(by_question):
            best = max(by_question[qidx], key=lambda x: x["score"])
            expert_pairs.append({"prompt": best["prompt"], "response": best["response"]})

        if not expert_pairs:
            raise RuntimeError("No expert pairs were generated.")

        train_ds = PromptResponseDataset(expert_pairs)
        loader = DataLoader(
            train_ds,
            batch_size=args.per_device_batch_size,
            shuffle=True,
            collate_fn=make_collate_fn(tokenizer, args.max_length),
        )

        for _ in range(sft_epochs):
            pbar = tqdm(loader, desc=f"ei_step={ei_step}", leave=False)
            micro_step = 0
            for batch_t in pbar:
                micro_step += 1
                input_ids = batch_t["input_ids"].to(policy_device)
                labels = batch_t["labels"].to(policy_device)
                response_mask = batch_t["response_mask"].to(policy_device)

                out = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )

                _, meta_loss = sft_microbatch_train_step(
                    policy_log_probs=out["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    normalize_constant=1.0,
                )

                if micro_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix(loss=float(meta_loss["normalized_loss"].detach().cpu()))

        policy.eval()
        val_acc = evaluate_accuracy(
            policy=policy,
            llm=llm,
            template=template,
            valid_data=valid_data,
            max_eval_examples=args.max_eval_examples,
            max_new_tokens=args.eval_max_new_tokens,
        )
        entropy = compute_entropy_on_dataset(policy, tokenizer, expert_pairs[: args.max_entropy_examples], args.max_length, policy_device)
        policy.train()

        acc_curve.append({"ei_step": ei_step, "validation_answer_accuracy": val_acc})
        ent_curve.append({"ei_step": ei_step, "response_entropy": entropy})
        logger.info("config=%s step=%d val_acc=%.4f entropy=%.4f", run_dir.name, ei_step, val_acc, entropy)

    with open(run_dir / "accuracy_curve.json", "w", encoding="utf-8") as f:
        json.dump(acc_curve, f, indent=2)
    with open(run_dir / "entropy_curve.json", "w", encoding="utf-8") as f:
        json.dump(ent_curve, f, indent=2)

    summary = {
        "rollout_g": rollout_g,
        "sft_epochs": sft_epochs,
        "db_size": db_size,
        "n_ei_steps": args.n_ei_steps,
        "accuracy_curve_path": str(run_dir / "accuracy_curve.json"),
        "entropy_curve_path": str(run_dir / "entropy_curve.json"),
        "final_validation_answer_accuracy": float(acc_curve[-1]["validation_answer_accuracy"]),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_int_csv(csv_str: str) -> list[int]:
    return [int(x.strip()) for x in csv_str.split(",") if x.strip()]


def build_config_grid(rollouts: list[int], epochs: list[int], db_sizes: list[int], limit: int) -> list[tuple[int, int, int]]:
    combos: list[tuple[int, int, int]] = []
    for g in rollouts:
        for e in epochs:
            for d in db_sizes:
                combos.append((g, e, d))
    if limit > 0:
        return combos[:limit]
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert iteration sweep for Qwen2.5-Math-1.5B")
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train-jsonl-path", type=str, default="/data/a5-alignment/MATH/train.jsonl")
    parser.add_argument("--validation-jsonl-path", type=str, default="/data/a5-alignment/MATH/validation.jsonl")
    parser.add_argument("--prompt-template-path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output-dir", type=str, default="outputs/ei_math_sweep")

    parser.add_argument("--n-ei-steps", type=int, default=5)
    parser.add_argument("--rollout-counts", type=str, default="4,8")
    parser.add_argument("--sft-epochs-list", type=str, default="1,2")
    parser.add_argument("--db-sizes", type=str, default="512,1024,2048")
    parser.add_argument("--max-configs", type=int, default=4)

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)

    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=1024)

    parser.add_argument("--max-eval-examples", type=int, default=256)
    parser.add_argument("--eval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-entropy-examples", type=int, default=64)

    parser.add_argument("--policy-device", type=str, default="cuda:0")
    parser.add_argument("--vllm-device", type=str, default="cuda:0")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.80)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rollouts = parse_int_csv(args.rollout_counts)
    epochs = parse_int_csv(args.sft_epochs_list)
    db_sizes = parse_int_csv(args.db_sizes)
    configs = build_config_grid(rollouts, epochs, db_sizes, args.max_configs)

    all_summaries: list[dict[str, Any]] = []
    for g, e, d in configs:
        run_name = f"g{g}_e{e}_db{d}"
        run_dir = output_dir / run_name
        summary = run_single_config(args, rollout_g=g, sft_epochs=e, db_size=d, run_dir=run_dir)
        summary["run_name"] = run_name
        all_summaries.append(summary)

    best = max(all_summaries, key=lambda x: x["final_validation_answer_accuracy"]) if all_summaries else None
    sweep_summary = {
        "configs_run": all_summaries,
        "best": best,
    }
    with open(output_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(sweep_summary, f, indent=2)

    logger.info("Saved EI sweep summary to %s", output_dir / "sweep_summary.json")


if __name__ == "__main__":
    main()
