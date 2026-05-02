import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    base = Path("outputs/sft_math_sweep")
    report_path = base / "sweep_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")

    report = load_json(report_path)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: uv add matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(9, 5))

    rows = []
    for run in report.get("dataset_size_runs", []):
        size_label = str(run["size"])
        curve_path = Path(run["curve_path"])
        curve = load_json(curve_path)
        if not curve:
            continue

        x = [int(p["step"]) for p in curve]
        y = [float(p["answer_accuracy"]) for p in curve]
        ax.plot(x, y, marker="o", linewidth=1.8, label=f"size={size_label}")

        rows.append(
            {
                "run": f"size={size_label}",
                "num_train_examples": int(run["num_train_examples"]),
                "final_answer_accuracy": float(run["final_answer_accuracy"]),
                "curve_path": str(curve_path),
            }
        )

    filtered = report.get("filtered_full_run")
    if filtered:
        fcurve_path = Path(filtered["curve_path"])
        fcurve = load_json(fcurve_path)
        if fcurve:
            x = [int(p["step"]) for p in fcurve]
            y = [float(p["answer_accuracy"]) for p in fcurve]
            ax.plot(x, y, marker="s", linewidth=2.2, linestyle="--", label="filtered_full")

        rows.append(
            {
                "run": "filtered_full",
                "num_train_examples": int(filtered["num_train_examples"]),
                "final_answer_accuracy": float(filtered["final_answer_accuracy"]),
                "curve_path": str(fcurve_path),
            }
        )

    ax.set_title("Qwen2.5-Math-1.5B SFT Validation Accuracy Curves")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Validation answer accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    png_path = base / "validation_accuracy_curves.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)

    table_path = base / "results_table.json"
    with table_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved plot: {png_path}")
    print(f"Saved table: {table_path}")
    print("\nSummary:")
    for r in rows:
        print(
            f"{r['run']:>14} | n={r['num_train_examples']:>6} | "
            f"final_acc={r['final_answer_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
