import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def summarize_baseline_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    total = 0
    cat1 = 0
    cat2 = 0
    cat3 = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            fr = safe_float(row.get("format_reward"), 0.0)
            ar = safe_float(row.get("answer_reward"), 0.0)
            if fr == 1.0 and ar == 1.0:
                cat1 += 1
            elif fr == 1.0 and ar == 0.0:
                cat2 += 1
            else:
                cat3 += 1

    if total == 0:
        return None

    return {
        "path": str(path),
        "total": total,
        "cat1": cat1,
        "cat2": cat2,
        "cat3": cat3,
        "answer_accuracy": cat1 / total,
        "format_accuracy": (cat1 + cat2) / total,
    }


def size_sort_key(size_label: str) -> tuple[int, int]:
    if size_label == "full":
        return (1, 10**9)
    try:
        return (0, int(size_label))
    except Exception:
        return (0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown report for SFT MATH sweep outputs")
    parser.add_argument(
        "--sweep-report",
        type=str,
        default="outputs/sft_math_sweep/sweep_report.json",
        help="Path to sweep_report.json",
    )
    parser.add_argument(
        "--baseline-jsonl",
        type=str,
        default="outputs/math_baseline/Qwen2.5-Math-1.5B_r1_zero.jsonl",
        help="Path to baseline evaluation jsonl (optional)",
    )
    parser.add_argument(
        "--curves-image",
        type=str,
        default="outputs/sft_math_sweep/validation_accuracy_curves.png",
        help="Path to curves image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sft_math_sweep/report.md",
        help="Path to write markdown report",
    )
    args = parser.parse_args()

    sweep_report_path = Path(args.sweep_report)
    if not sweep_report_path.exists():
        raise FileNotFoundError(f"Missing sweep report: {sweep_report_path}")

    report = load_json(sweep_report_path)
    if not isinstance(report, dict):
        raise ValueError("sweep_report.json is not a JSON object")

    dataset_runs = list(report.get("dataset_size_runs", []))
    dataset_runs.sort(key=lambda x: size_sort_key(str(x.get("size", ""))))

    full_run = next((r for r in dataset_runs if str(r.get("size")) == "full"), None)
    full_acc = safe_float((full_run or {}).get("final_answer_accuracy"), 0.0)
    meets_target = full_acc >= 0.15

    filtered = report.get("filtered_full_run") or {}
    filtered_n = int(filtered.get("num_train_examples") or 0)
    filtered_acc = safe_float(filtered.get("final_answer_accuracy"), 0.0)
    filter_stats = filtered.get("filter_stats") or {}

    comparison = report.get("comparison") or {}
    delta = safe_float(comparison.get("delta_filtered_minus_unfiltered"), filtered_acc - full_acc)

    baseline_summary = summarize_baseline_jsonl(Path(args.baseline_jsonl))

    curves_image = Path(args.curves_image)

    lines: list[str] = []
    lines.append("# SFT MATH Experiment Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Model: Qwen/Qwen2.5-Math-1.5B")
    lines.append("- Train data: /data/a5-alignment/MATH/sft.jsonl")
    lines.append("- Validation data: /data/a5-alignment/MATH/validation.jsonl")
    lines.append("- Sweep sizes: 128, 256, 512, 1024, full")
    lines.append("")

    lines.append("## Deliverable A: Validation Accuracy Curves by Dataset Size")
    if curves_image.exists():
        lines.append(f"![Validation accuracy curves]({curves_image.as_posix()})")
    else:
        lines.append("- Curves image not found. Run scripts/plot_sft_math_curves.py after sweep completion.")
    lines.append("")

    lines.append("| Train size | Examples used | Final validation answer accuracy |")
    lines.append("| --- | ---: | ---: |")
    for run in dataset_runs:
        size = str(run.get("size", "?"))
        n = int(run.get("num_train_examples") or 0)
        acc = safe_float(run.get("final_answer_accuracy"), 0.0)
        lines.append(f"| {size} | {n} | {pct(acc)} |")
    lines.append("")

    lines.append("## Full Dataset Target Check")
    lines.append(f"- Full-dataset final validation answer accuracy: {pct(full_acc)}")
    lines.append(f"- Meets target (>= 15%): {'Yes' if meets_target else 'No'}")
    lines.append("")

    lines.append("## Deliverable B: Filtered-Correct SFT Run")
    lines.append(f"- Filtered dataset size: {filtered_n}")
    lines.append(f"- Filtered full-dataset final validation answer accuracy: {pct(filtered_acc)}")
    if filter_stats:
        lines.append("- Filter stats:")
        lines.append(f"  - Total examples before filtering: {int(filter_stats.get('total_examples', 0))}")
        lines.append(f"  - Examples with ground truth: {int(filter_stats.get('with_ground_truth', 0))}")
        lines.append(f"  - Correct examples kept: {int(filter_stats.get('correct_examples', 0))}")
        lines.append(
            f"  - Dropped missing ground truth: {int(filter_stats.get('dropped_missing_ground_truth', 0))}"
        )
        lines.append(f"  - Dropped incorrect: {int(filter_stats.get('dropped_incorrect', 0))}")
    lines.append("")

    lines.append("## Comparison to Previous Experiment")
    if baseline_summary is not None:
        lines.append(
            f"- Previous baseline answer accuracy (Category 1 / total): {pct(baseline_summary['answer_accuracy'])}"
        )
        lines.append(
            f"- Previous baseline format accuracy ((Category 1 + 2) / total): {pct(baseline_summary['format_accuracy'])}"
        )
        lines.append(
            f"- Previous baseline counts: C1={baseline_summary['cat1']}, "
            f"C2={baseline_summary['cat2']}, C3={baseline_summary['cat3']}, total={baseline_summary['total']}"
        )
    else:
        lines.append("- Baseline JSONL not found; baseline comparison could not be computed automatically.")

    lines.append(f"- Filtered minus unfiltered full-data delta: {pct(delta)}")
    lines.append("")

    lines.append("## Artifact Paths")
    lines.append(f"- Sweep report: {sweep_report_path.as_posix()}")
    lines.append(f"- Curves image: {curves_image.as_posix()}")
    lines.append(f"- Markdown report: {Path(args.output).as_posix()}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote report: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
