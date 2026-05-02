import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


DEFAULT_MODEL = os.environ.get("QWEN_BASE_PATH", "Qwen/Qwen2.5-Math-1.5B")
DEFAULT_DATA = os.environ.get("DATA_PATH", "/content/validation.jsonl")
DEFAULT_PROMPT = os.environ.get("PROMPT_PATH", "prompts/r1_zero.prompt")
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/content/outputs/math_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen-Math baseline with r1_zero reward.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path or HF model id.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to validation .jsonl file.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt template path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to write output jsonl.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap for quick Colab smoke tests.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def resolve_prompt_path(prompt_path: str) -> str:
    prompt_candidate = Path(prompt_path)
    if prompt_candidate.is_file():
        return str(prompt_candidate)

    file_relative = Path(__file__).resolve().parent / prompt_path
    if file_relative.is_file():
        return str(file_relative)

    raise FileNotFoundError(
        f"Prompt template not found. Tried: '{prompt_path}' and '{file_relative}'"
    )


# Core vLLM generation
def run_vllm(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    outputs = llm.generate(prompts, sampling_params)
    return [out.text for resp in outputs for out in resp.outputs]


# Evaluation
def evaluate_vllm(
    llm: LLM,
    reward_fn: Callable[[str, str], dict],
    prompts: List[str],
    answers: List[str],
    sampling_params: SamplingParams,
) -> List[dict]:
    responses = run_vllm(llm, prompts, sampling_params)

    results = []
    for prompt, response, answer in zip(prompts, responses, answers):
        reward_dict = reward_fn(response, answer)
        results.append(
            {
                "prompt": prompt,
                "response": response,
                "answer": answer,
                "format_reward": reward_dict["format_reward"],
                "answer_reward": reward_dict["answer_reward"],
            }
        )

    return results


# Data loading
def load_and_format_prompts(
    data_path: str,
    prompt_path: str,
    max_examples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    resolved_prompt_path = resolve_prompt_path(prompt_path)

    with open(resolved_prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompts, answers = [], []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(prompt_template.format(question=data["problem"]))
            answers.append(data["answer"])
            if max_examples is not None and len(prompts) >= max_examples:
                break

    return prompts, answers


# Model + sampling
def build_llm(model_path: str, tensor_parallel_size: int) -> LLM:
    return LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)


def build_sampling_params(temperature: float, top_p: float, max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


# Metrics + inspection
def compute_metrics(results: List[dict]) -> None:
    total = len(results)
    if total == 0:
        print("No results to score.")
        return

    counter = Counter()
    bad_formats = []
    bad_answers = []

    for r in results:
        if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0:
            counter["cat1_correct_both_1"] += 1
        elif r["format_reward"] == 1.0 and r["answer_reward"] == 0.0:
            counter["cat2_format1_answer0"] += 1
            bad_answers.append(r["response"])
        else:
            counter["cat3_format0_answer0"] += 1
            bad_formats.append(r["response"])

    accuracy = counter["cat1_correct_both_1"] / total
    format_acc = (counter["cat1_correct_both_1"] + counter["cat2_format1_answer0"]) / total

    print("\n=== Metrics ===")
    print(f"Total examples: {total}")
    print(f"Category (1) format=1 answer=1: {counter['cat1_correct_both_1']}")
    print(f"Category (2) format=1 answer=0: {counter['cat2_format1_answer0']}")
    print(f"Category (3) format=0 answer=0: {counter['cat3_format0_answer0']}")
    print(f"Accuracy (category 1): {accuracy:.4f}")
    print(f"Format accuracy (cat1 + cat2): {format_acc:.4f}")

    print("\n--- Example bad formats (first 10) ---")
    for x in bad_formats[:10]:
        print(x)
        print("-" * 40)

    print("\n--- Example wrong answers (first 10) ---")
    for x in bad_answers[:10]:
        print(x)
        print("-" * 40)


# Save results
def save_results(results: List[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model.rstrip("/").split("/")[-1]
    output_path = output_dir / f"{model_name}_r1_zero.jsonl"

    print("Loading data...")
    prompts, answers = load_and_format_prompts(args.data, args.prompt, args.max_examples)
    print(f"Loaded {len(prompts)} examples")

    print("Building model...")
    llm = build_llm(args.model, args.tensor_parallel_size)
    sampling_params = build_sampling_params(args.temperature, args.top_p, args.max_tokens)

    print("Running evaluation...")
    results = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        answers,
        sampling_params,
    )

    print("Computing metrics...")
    compute_metrics(results)

    print("Saving results...")
    save_results(results, str(output_path))

    print(f"\nDone! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
