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
import wandb
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train import sft_microbatch_train_step
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_JSONL = "/data/a5-alignment/MATH/sft.jsonl"
DEFAULT_VALID_JSONL = "/data/a5-alignment/MATH/validation.jsonl"
DEFAULT_PROMPT_TEMPLATE = "cs336_alignment/prompts/r1_zero.prompt"


def run_vllm(vllm_model, prompts: list[str], sampling_params) -> list[str]:
	outputs = vllm_model.generate(prompts, sampling_params)
	return [output.text for response in outputs for output in response.outputs]


def evaluate_vllm(
	vllm_model,
	reward_fn,
	prompts: list[str],
	answers: list[str],
	eval_sampling_params,
) -> list[dict[str, Any]]:
	responses = run_vllm(vllm_model, prompts, eval_sampling_params)
	info_dicts: list[dict[str, Any]] = []

	for prompt, answer, response in zip(prompts, answers, responses):
		metrics = reward_fn(response, answer)
		info_dicts.append(
			{
				"prompt": prompt,
				"answer": answer,
				"response": response,
				"format_reward": float(metrics["format_reward"]),
				"answer_reward": float(metrics["answer_reward"]),
				"reward": float(metrics["reward"]),
			}
		)

	return info_dicts


class PromptResponseDataset(Dataset):
	def __init__(self, examples: list[dict[str, Any]]) -> None:
		self.examples = examples

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, idx: int) -> dict[str, Any]:
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


def normalize_to_prompt_response(row: dict[str, Any]) -> dict[str, Any]:
	prompt = row.get("prompt") or row.get("question") or row.get("problem") or row.get("input")
	response = (
		row.get("response")
		or row.get("solution")
		or row.get("output")
		or row.get("completion")
		or row.get("answer")
	)
	ground_truth = row.get("ground_truth") or row.get("final_answer") or row.get("answer") or row.get("label")

	if prompt is None or response is None:
		raise ValueError(f"Could not map row to prompt/response fields: keys={list(row.keys())}")

	return {
		"prompt": str(prompt),
		"response": str(response),
		"ground_truth": str(ground_truth) if ground_truth is not None else None,
	}


def load_train_examples(train_jsonl_path: str | None, hf_dataset_name: str) -> list[dict[str, Any]]:
	if train_jsonl_path and Path(train_jsonl_path).exists():
		logger.info("Loading SFT train data from jsonl: %s", train_jsonl_path)
		rows = read_jsonl(train_jsonl_path)
		return [normalize_to_prompt_response(r) for r in rows]

	logger.info("Loading SFT train data from Hugging Face: %s", hf_dataset_name)
	ds = load_dataset(hf_dataset_name)
	if "train" not in ds:
		raise ValueError(f"Dataset {hf_dataset_name} has no train split.")
	return [normalize_to_prompt_response(r) for r in ds["train"]]


def filter_examples_with_correct_answer(
	examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
	filtered: list[dict[str, Any]] = []
	stats = {
		"total_examples": len(examples),
		"with_ground_truth": 0,
		"correct_examples": 0,
		"dropped_missing_ground_truth": 0,
		"dropped_incorrect": 0,
	}

	for ex in examples:
		ground_truth = ex.get("ground_truth")
		if ground_truth is None:
			stats["dropped_missing_ground_truth"] += 1
			continue

		stats["with_ground_truth"] += 1
		response = ex["response"]
		metrics = question_only_reward_fn(response, str(ground_truth))
		if float(metrics.get("answer_reward", 0.0)) == 1.0:
			filtered.append(ex)
			stats["correct_examples"] += 1
		else:
			stats["dropped_incorrect"] += 1

	return filtered, stats


def maybe_write_filtered_examples(path: str | None, examples: list[dict[str, Any]]) -> None:
	if not path:
		return
	out_path = Path(path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		for ex in examples:
			f.write(
				json.dumps({"prompt": ex["prompt"], "response": ex["response"]}, ensure_ascii=False)
				+ "\n"
			)


def load_validation_problem_answer(validation_jsonl_path: str) -> tuple[list[str], list[str]]:
	rows = read_jsonl(validation_jsonl_path)
	problems: list[str] = []
	answers: list[str] = []

	for row in rows:
		problem = row.get("problem") or row.get("question")
		answer = row.get("answer") or row.get("solution")
		if problem is None or answer is None:
			continue
		problems.append(str(problem))
		answers.append(str(answer))

	if not problems:
		raise ValueError(f"No usable validation examples in {validation_jsonl_path}")

	return problems, answers


def resolve_prompt_template(prompt_template_path: str) -> str:
	candidate = Path(prompt_template_path)
	if candidate.exists():
		return candidate.read_text(encoding="utf-8")

	repo_root = Path(__file__).resolve().parents[1]
	repo_candidate = repo_root / prompt_template_path
	if repo_candidate.exists():
		return repo_candidate.read_text(encoding="utf-8")

	raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")


def build_eval_prompts(validation_jsonl_path: str, prompt_template_path: str) -> tuple[list[str], list[str]]:
	template = resolve_prompt_template(prompt_template_path)
	questions, answers = load_validation_problem_answer(validation_jsonl_path)
	prompts = [template.format(question=q) for q in questions]
	return prompts, answers


def make_collate_fn(tokenizer, max_length: int):
	def collate(examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
		prompt_strs = [x["prompt"] for x in examples]
		output_strs = [x["response"] for x in examples]

		tokenized = tokenize_prompt_and_output(
			prompt_strs=prompt_strs,
			output_strs=output_strs,
			tokenizer=tokenizer,
		)

		input_ids = tokenized["input_ids"][:, :max_length]
		labels = tokenized["labels"][:, :max_length]
		response_mask = tokenized["response_mask"][:, :max_length]

		return {
			"input_ids": input_ids,
			"labels": labels,
			"response_mask": response_mask,
		}

	return collate


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


@torch.no_grad()
def evaluate_with_vllm(
	policy,
	llm,
	prompts: list[str],
	answers: list[str],
	max_eval_examples: int,
	max_tokens: int,
) -> dict[str, float]:
	vllm_module = importlib.import_module("vllm")
	SamplingParams = getattr(vllm_module, "SamplingParams")

	if max_eval_examples is not None and max_eval_examples > 0:
		prompts = prompts[:max_eval_examples]
		answers = answers[:max_eval_examples]

	load_policy_into_vllm_instance(policy, llm)
	sampling_params = SamplingParams(
		temperature=0.0,
		top_p=1.0,
		max_tokens=max_tokens,
		stop=["</answer>"],
		include_stop_str_in_output=True,
	)

	records = evaluate_vllm(
		vllm_model=llm,
		reward_fn=r1_zero_reward_fn,
		prompts=prompts,
		answers=answers,
		eval_sampling_params=sampling_params,
	)

	if not records:
		return {"answer_accuracy": 0.0, "format_accuracy": 0.0, "reward_mean": 0.0}

	answer_accuracy = sum(float(r["answer_reward"]) for r in records) / len(records)
	format_accuracy = sum(float(r["format_reward"]) for r in records) / len(records)
	reward_mean = sum(float(r["reward"]) for r in records) / len(records)
	return {
		"answer_accuracy": answer_accuracy,
		"format_accuracy": format_accuracy,
		"reward_mean": reward_mean,
	}


def train(args: argparse.Namespace) -> dict[str, Any]:
	set_seed(args.seed)
	os.makedirs(args.output_dir, exist_ok=True)

	train_examples = load_train_examples(args.train_jsonl_path, args.hf_dataset_name)
	filter_stats: dict[str, int] | None = None
	if args.filter_correct_only:
		train_examples, filter_stats = filter_examples_with_correct_answer(train_examples)
		maybe_write_filtered_examples(args.filtered_examples_output_path, train_examples)
		logger.info("Filtered correct-only examples: %d", len(train_examples))
		logger.info("Filter stats: %s", filter_stats)

	if args.max_train_examples is not None:
		train_examples = train_examples[: args.max_train_examples]

	logger.info("Loaded %d SFT training examples", len(train_examples))

	eval_prompts, eval_answers = build_eval_prompts(args.validation_jsonl_path, args.prompt_template_path)
	logger.info("Loaded %d validation examples", len(eval_prompts))

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

	if args.gradient_checkpointing:
		policy.gradient_checkpointing_enable()
	else:
		policy.gradient_checkpointing_disable()

	train_ds = PromptResponseDataset(train_examples)
	train_loader = DataLoader(
		train_ds,
		batch_size=args.per_device_batch_size,
		shuffle=True,
		collate_fn=make_collate_fn(tokenizer, args.max_length),
		drop_last=False,
	)

	updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
	if args.max_train_steps is not None:
		total_update_steps = args.max_train_steps
	else:
		total_update_steps = args.num_train_epochs * updates_per_epoch

	warmup_steps = int(args.warmup_ratio * total_update_steps)
	optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

	def lr_lambda(step: int) -> float:
		if total_update_steps <= 0:
			return 1.0
		if warmup_steps > 0 and step < warmup_steps:
			return float(step) / float(max(1, warmup_steps))
		progress = (step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
		return max(0.0, 1.0 - progress)

	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

	use_wandb = not args.disable_wandb
	if use_wandb:
		wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
		wandb.define_metric("train_step")
		wandb.define_metric("eval_step")
		wandb.define_metric("train/*", step_metric="train_step")
		wandb.define_metric("eval/*", step_metric="eval_step")

	vllm_instance = None
	if args.use_vllm_eval:
		vllm_instance = init_vllm(
			model_id=args.model_name_or_path,
			device=args.vllm_device,
			seed=args.seed,
			gpu_memory_utilization=args.vllm_gpu_memory_utilization,
		)

	global_update_step = 0
	micro_step = 0
	running_loss = 0.0
	curve_rows: list[dict[str, float | int]] = []

	optimizer.zero_grad(set_to_none=True)

	for epoch in range(args.num_train_epochs):
		epoch_iterator = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
		for batch in epoch_iterator:
			micro_step += 1
			input_ids = batch["input_ids"].to(policy_device)
			labels = batch["labels"].to(policy_device)
			response_mask = batch["response_mask"].to(policy_device)

			out = get_response_log_probs(
				model=policy,
				input_ids=input_ids,
				labels=labels,
				return_token_entropy=False,
			)

			_, meta = sft_microbatch_train_step(
				policy_log_probs=out["log_probs"],
				response_mask=response_mask,
				gradient_accumulation_steps=args.gradient_accumulation_steps,
				normalize_constant=1.0,
			)
			running_loss += float(meta["normalized_loss"].detach().cpu())

			should_step = (micro_step % args.gradient_accumulation_steps == 0)
			if should_step:
				torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad(set_to_none=True)

				global_update_step += 1
				mean_train_loss = running_loss / args.gradient_accumulation_steps
				running_loss = 0.0

				if use_wandb:
					wandb.log(
						{
							"train_step": global_update_step,
							"train/loss": mean_train_loss,
							"train/lr": float(scheduler.get_last_lr()[0]),
						}
					)

				if global_update_step % args.log_every_steps == 0:
					logger.info(
						"step=%d loss=%.4f lr=%.2e",
						global_update_step,
						mean_train_loss,
						scheduler.get_last_lr()[0],
					)

				if (
					args.eval_every_steps > 0
					and vllm_instance is not None
					and global_update_step % args.eval_every_steps == 0
				):
					policy.eval()
					eval_metrics = evaluate_with_vllm(
						policy=policy,
						llm=vllm_instance,
						prompts=eval_prompts,
						answers=eval_answers,
						max_eval_examples=args.max_eval_examples,
						max_tokens=args.eval_max_new_tokens,
					)
					policy.train()

					row = {
						"step": global_update_step,
						"answer_accuracy": eval_metrics["answer_accuracy"],
						"format_accuracy": eval_metrics["format_accuracy"],
						"reward_mean": eval_metrics["reward_mean"],
					}
					curve_rows.append(row)
					logger.info(
						"eval step=%d answer_acc=%.4f format_acc=%.4f reward=%.4f",
						global_update_step,
						row["answer_accuracy"],
						row["format_accuracy"],
						row["reward_mean"],
					)

					if use_wandb:
						wandb.log(
							{
								"eval_step": global_update_step,
								"eval/answer_accuracy": row["answer_accuracy"],
								"eval/format_accuracy": row["format_accuracy"],
								"eval/reward_mean": row["reward_mean"],
							}
						)

				if global_update_step >= total_update_steps:
					break

		if global_update_step >= total_update_steps:
			break

	save_dir = Path(args.output_dir)
	save_dir.mkdir(parents=True, exist_ok=True)
	policy.save_pretrained(save_dir / "policy")
	tokenizer.save_pretrained(save_dir / "policy")

	curve_path = save_dir / "validation_curve.json"
	with open(curve_path, "w", encoding="utf-8") as f:
		json.dump(curve_rows, f, indent=2)

	summary = {
		"num_train_examples": len(train_examples),
		"total_update_steps": global_update_step,
		"final_learning_rate": float(scheduler.get_last_lr()[0]),
		"curve_path": str(curve_path),
		"last_eval": curve_rows[-1] if curve_rows else None,
	}
	summary_path = save_dir / "summary.json"
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	logger.info("Saved model to %s", save_dir / "policy")
	logger.info("Saved validation curve to %s", curve_path)
	logger.info("Saved summary to %s", summary_path)

	if use_wandb:
		wandb.finish()

	return summary


def clone_args(args: argparse.Namespace) -> argparse.Namespace:
	return argparse.Namespace(**vars(args).copy())


def parse_int_csv(csv_str: str) -> list[int]:
	items = [x.strip() for x in csv_str.split(",") if x.strip()]
	return [int(x) for x in items]


def parse_float_csv(csv_str: str) -> list[float]:
	items = [x.strip() for x in csv_str.split(",") if x.strip()]
	return [float(x) for x in items]


def run_full_dataset_tuning(args: argparse.Namespace) -> dict[str, Any]:
	learning_rates = parse_float_csv(args.tune_learning_rates)
	accum_steps = parse_int_csv(args.tune_accum_steps)

	results: list[dict[str, Any]] = []
	base_out = Path(args.output_dir) / "tuning"

	for lr in learning_rates:
		for ga in accum_steps:
			run_args = clone_args(args)
			run_args.learning_rate = lr
			run_args.gradient_accumulation_steps = ga
			run_args.max_train_examples = None
			run_args.filter_correct_only = False
			run_args.output_dir = str(base_out / f"lr_{lr:g}_ga_{ga}")
			if run_args.run_name is not None:
				run_args.run_name = f"{run_args.run_name}_lr{lr:g}_ga{ga}"

			summary = train(run_args)
			last_eval = summary.get("last_eval") or {}
			results.append(
				{
					"learning_rate": lr,
					"gradient_accumulation_steps": ga,
					"summary_path": str(Path(run_args.output_dir) / "summary.json"),
					"curve_path": str(Path(run_args.output_dir) / "validation_curve.json"),
					"final_answer_accuracy": float(last_eval.get("answer_accuracy", 0.0)),
				}
			)

	best = max(results, key=lambda r: r["final_answer_accuracy"]) if results else None
	return {"grid_results": results, "best": best}


def run_sweep_suite(args: argparse.Namespace) -> dict[str, Any]:
	sweep_sizes = parse_int_csv(args.sweep_sizes)
	base_output = Path(args.output_dir)
	base_output.mkdir(parents=True, exist_ok=True)

	report: dict[str, Any] = {
		"dataset_size_runs": [],
		"full_dataset_tuning": None,
		"filtered_full_run": None,
		"comparison": {},
	}

	if args.run_full_tuning:
		tuning_args = clone_args(args)
		tuning_args.output_dir = str(base_output)
		tuning_args.run_size_sweep = False
		tuning_args.run_filtered_experiment = False
		report["full_dataset_tuning"] = run_full_dataset_tuning(tuning_args)
		best = report["full_dataset_tuning"]["best"]
		if best is not None:
			args.learning_rate = best["learning_rate"]
			args.gradient_accumulation_steps = best["gradient_accumulation_steps"]

	for size in sweep_sizes + [-1]:
		run_args = clone_args(args)
		run_args.run_size_sweep = False
		run_args.run_filtered_experiment = False
		run_args.filter_correct_only = False
		label = "full" if size == -1 else str(size)
		run_args.max_train_examples = None if size == -1 else size
		run_args.output_dir = str(base_output / f"size_{label}")
		if run_args.run_name is not None:
			run_args.run_name = f"{run_args.run_name}_size_{label}"

		summary = train(run_args)
		report["dataset_size_runs"].append(
			{
				"size": label,
				"num_train_examples": summary["num_train_examples"],
				"summary_path": str(Path(run_args.output_dir) / "summary.json"),
				"curve_path": str(Path(run_args.output_dir) / "validation_curve.json"),
				"final_answer_accuracy": float((summary.get("last_eval") or {}).get("answer_accuracy", 0.0)),
			}
		)

	if args.run_filtered_experiment:
		filtered_args = clone_args(args)
		filtered_args.run_size_sweep = False
		filtered_args.run_filtered_experiment = False
		filtered_args.max_train_examples = None
		filtered_args.filter_correct_only = True
		filtered_args.output_dir = str(base_output / "filtered_full")
		filtered_args.filtered_examples_output_path = str(base_output / "filtered_full" / "filtered_sft.jsonl")
		if filtered_args.run_name is not None:
			filtered_args.run_name = f"{filtered_args.run_name}_filtered_full"

		filtered_summary = train(filtered_args)
		report["filtered_full_run"] = {
			"num_train_examples": filtered_summary["num_train_examples"],
			"summary_path": str(Path(filtered_args.output_dir) / "summary.json"),
			"curve_path": str(Path(filtered_args.output_dir) / "validation_curve.json"),
			"final_answer_accuracy": float((filtered_summary.get("last_eval") or {}).get("answer_accuracy", 0.0)),
			"filter_stats": filtered_summary.get("filter_stats"),
		}

	full_unfiltered = next((r for r in report["dataset_size_runs"] if r["size"] == "full"), None)
	filtered = report.get("filtered_full_run")
	if full_unfiltered is not None and filtered is not None:
		report["comparison"] = {
			"full_unfiltered_final_answer_accuracy": full_unfiltered["final_answer_accuracy"],
			"filtered_full_final_answer_accuracy": filtered["final_answer_accuracy"],
			"delta_filtered_minus_unfiltered": (
				filtered["final_answer_accuracy"] - full_unfiltered["final_answer_accuracy"]
			),
			"filtered_dataset_size": filtered["num_train_examples"],
		}

	report_path = base_output / "sweep_report.json"
	with open(report_path, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)
	logger.info("Saved sweep report to %s", report_path)

	return report


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run SFT for Qwen2.5-Math on MATH data")

	parser.add_argument("--model-name-or-path", type=str, default=DEFAULT_MODEL)
	parser.add_argument("--hf-dataset-name", type=str, default="qwedsacf/competition_math")
	parser.add_argument("--train-jsonl-path", type=str, default=DEFAULT_TRAIN_JSONL)
	parser.add_argument("--validation-jsonl-path", type=str, default=DEFAULT_VALID_JSONL)
	parser.add_argument("--prompt-template-path", type=str, default=DEFAULT_PROMPT_TEMPLATE)
	parser.add_argument("--output-dir", type=str, default="outputs/sft_experiment")

	parser.add_argument("--learning-rate", type=float, default=1e-5)
	parser.add_argument("--weight-decay", type=float, default=0.01)
	parser.add_argument("--warmup-ratio", type=float, default=0.03)
	parser.add_argument("--max-grad-norm", type=float, default=1.0)

	parser.add_argument("--per-device-batch-size", type=int, default=1)
	parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
	parser.add_argument("--num-train-epochs", type=int, default=1)
	parser.add_argument("--max-train-steps", type=int, default=300)
	parser.add_argument("--max-length", type=int, default=1024)
	parser.add_argument("--max-train-examples", type=int, default=None)
	parser.add_argument("--filter-correct-only", action="store_true")
	parser.add_argument("--filtered-examples-output-path", type=str, default=None)

	parser.add_argument("--policy-device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--use-vllm-eval", action="store_true")
	parser.add_argument("--vllm-device", type=str, default="cuda:1")
	parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)

	parser.add_argument("--eval-every-steps", type=int, default=25)
	parser.add_argument("--max-eval-examples", type=int, default=256)
	parser.add_argument("--eval-max-new-tokens", type=int, default=1024)

	parser.add_argument("--log-every-steps", type=int, default=5)
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--gradient-checkpointing", action="store_true")

	parser.add_argument("--disable-wandb", action="store_true")
	parser.add_argument("--wandb-project", type=str, default="ece405-sft")
	parser.add_argument("--run-name", type=str, default=None)

	parser.add_argument("--run-size-sweep", action="store_true")
	parser.add_argument("--sweep-sizes", type=str, default="128,256,512,1024")
	parser.add_argument("--run-full-tuning", action="store_true")
	parser.add_argument("--tune-learning-rates", type=str, default="5e-6,1e-5,2e-5")
	parser.add_argument("--tune-accum-steps", type=str, default="8,16,32")
	parser.add_argument("--run-filtered-experiment", action="store_true")

	return parser.parse_args()


if __name__ == "__main__":
	logging.basicConfig(
		format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
		level=logging.INFO,
	)
	args = parse_args()
	if args.run_size_sweep:
		run_sweep_suite(args)
	else:
		train(args)