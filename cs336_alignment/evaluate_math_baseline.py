import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable, List, Tuple

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)

QWEN_BASE_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
DEFAULT_DATA_PATH = "/data/a5-alignment/MATH/validation.jsonl"
DEFAULT_PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_OUTPUT_DIR = "outputs/math_baseline"


def run_vllm(vllm_model: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
	outputs = vllm_model.generate(prompts, sampling_params)
	return [output.text for response in outputs for output in response.outputs]


def evaluate_vllm(
	vllm_model: LLM,
	reward_fn: Callable[[str, str], dict[str, float]],
	prompts: List[str],
	answers: List[str],
	eval_sampling_params: SamplingParams,
) -> List[dict[str, Any]]:
	"""
	Evaluate a language model on a list of prompts,
	compute evaluation metrics, and return per-example records.
	"""
	responses = run_vllm(vllm_model, prompts, eval_sampling_params)
	info_dicts: List[dict[str, Any]] = []

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


def _resolve_prompt_path(prompt_path: str) -> Path:
	candidate = Path(prompt_path)
	if candidate.exists():
		return candidate

	repo_root = Path(__file__).resolve().parents[1]
	repo_candidate = repo_root / prompt_path
	if repo_candidate.exists():
		return repo_candidate

	raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def load_and_format_prompts(data_path: str, prompt_path: str) -> Tuple[List[str], List[str]]:
	resolved_prompt_path = _resolve_prompt_path(prompt_path)
	with open(resolved_prompt_path, "r") as prompt_file:
		prompt_template = prompt_file.read()

	prompts: List[str] = []
	answers: List[str] = []

	with open(data_path, "r") as data_file:
		for line in data_file:
			data = json.loads(line)
			prompts.append(prompt_template.format(question=data["problem"]))
			answers.append(str(data["answer"]))

	return prompts, answers


def build_llm_and_params(model_path: str, num_gpus: int, max_tokens: int) -> Tuple[LLM, SamplingParams]:
	llm = LLM(model=model_path, tensor_parallel_size=num_gpus, trust_remote_code=True)
	sampling_params = SamplingParams(
		temperature=0.0,
		top_p=1.0,
		max_tokens=max_tokens,
		stop=["</answer>"],
		include_stop_str_in_output=True,
	)
	return llm, sampling_params


def serialize_results(records: List[dict[str, Any]], output_path: str) -> None:
	out_path = Path(output_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	with open(out_path, "w") as f:
		for record in records:
			f.write(json.dumps(record, ensure_ascii=False) + "\n")


def inspect_info_dicts(info_dicts: List[dict[str, Any]]) -> None:
	counter = Counter()
	bad_formats: List[str] = []
	bad_answers: List[str] = []

	for info_dict in info_dicts:
		format_reward = info_dict["format_reward"]
		answer_reward = info_dict["answer_reward"]

		if format_reward == 1.0 and answer_reward == 1.0:
			counter["correct"] += 1
		elif format_reward == 1.0 and answer_reward == 0.0:
			counter["format_correct_answer_incorrect"] += 1
			bad_answers.append(info_dict["response"])
		else:
			counter["incorrect"] += 1
			bad_formats.append(info_dict["response"])

	logger.info("Breakdown: %s", dict(counter))
	logger.info("Sample bad format responses (up to 10): %s", bad_formats[:10])
	logger.info("Sample bad answer responses (up to 10): %s", bad_answers[:10])


def summarize_records(records: List[dict[str, Any]], n_examples_to_show: int = 10) -> dict[str, Any]:
	counter = Counter()
	bad_formats: List[dict[str, Any]] = []
	bad_answers: List[dict[str, Any]] = []
	correct: List[dict[str, Any]] = []

	for record in records:
		format_reward = record["format_reward"]
		answer_reward = record["answer_reward"]

		if format_reward == 1.0 and answer_reward == 1.0:
			counter["correct"] += 1
			if len(correct) < n_examples_to_show:
				correct.append(record)
		elif format_reward == 1.0 and answer_reward == 0.0:
			counter["format_correct_answer_incorrect"] += 1
			if len(bad_answers) < n_examples_to_show:
				bad_answers.append(record)
		elif format_reward == 0.0 and answer_reward == 0.0:
			counter["format_incorrect_answer_incorrect"] += 1
			if len(bad_formats) < n_examples_to_show:
				bad_formats.append(record)

	summary = {
		"num_examples": len(records),
		"reward_mean": mean(record["reward"] for record in records) if records else 0.0,
		"format_reward_mean": mean(record["format_reward"] for record in records) if records else 0.0,
		"answer_reward_mean": mean(record["answer_reward"] for record in records) if records else 0.0,
		"category_counts": {
			"correct": counter["correct"],
			"format_correct_answer_incorrect": counter["format_correct_answer_incorrect"],
			"format_incorrect_answer_incorrect": counter["format_incorrect_answer_incorrect"],
		},
		"examples": {
			"correct": correct,
			"format_incorrect_answer_incorrect": bad_formats,
			"format_correct_answer_incorrect": bad_answers,
		},
	}
	return summary


def main(args: argparse.Namespace) -> None:
	prompts, answers = load_and_format_prompts(args.data_path, args.prompt_path)

	if args.max_examples is not None:
		prompts = prompts[: args.max_examples]
		answers = answers[: args.max_examples]

	llm, sampling_params = build_llm_and_params(args.model_path, args.num_gpus, args.max_tokens)
	records = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)
	inspect_info_dicts(records)

	serialize_results(records, args.output_path)
	summary = summarize_records(records, n_examples_to_show=args.examples_per_category)

	summary_path = Path(args.summary_path)
	summary_path.parent.mkdir(parents=True, exist_ok=True)
	with open(summary_path, "w") as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	logger.info("Saved per-example results to %s", args.output_path)
	logger.info("Saved summary to %s", args.summary_path)
	logger.info("Category counts: %s", summary["category_counts"])
	logger.info(
		"Metrics: reward_mean=%.4f format_reward_mean=%.4f answer_reward_mean=%.4f",
		summary["reward_mean"],
		summary["format_reward_mean"],
		summary["answer_reward_mean"],
	)


if __name__ == "__main__":
	logging.basicConfig(
		format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
		level=logging.INFO,
	)

	parser = argparse.ArgumentParser(description="Evaluate zero-shot Qwen2.5-Math-1.5B on MATH validation")
	parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
	parser.add_argument("--model-path", type=str, default=QWEN_BASE_PATH)
	parser.add_argument("--prompt-path", type=str, default=DEFAULT_PROMPT_PATH)
	parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--output-path", type=str, default=None)
	parser.add_argument("--summary-path", type=str, default=None)
	parser.add_argument("--num-gpus", type=int, default=1)
	parser.add_argument("--max-tokens", type=int, default=1024)
	parser.add_argument("--max-examples", type=int, default=None)
	parser.add_argument("--examples-per-category", type=int, default=10)
	parsed_args = parser.parse_args()

	out_dir = Path(parsed_args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	model_name = Path(parsed_args.model_path).name
	if parsed_args.output_path is None:
		parsed_args.output_path = str(out_dir / f"{model_name}_r1_zero.jsonl")
	if parsed_args.summary_path is None:
		parsed_args.summary_path = str(out_dir / f"{model_name}_r1_zero_summary.json")

	logger.info("Running evaluator")
	main(parsed_args)
