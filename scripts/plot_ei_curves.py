import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    base = Path("outputs/ei_math_sweep")
    sweep = base / "sweep_summary.json"
    if not sweep.exists():
        raise FileNotFoundError(f"Missing sweep summary: {sweep}")

    report = load_json(sweep)
    runs = report.get("configs_run", [])

    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    fig2, ax2 = plt.subplots(figsize=(9, 5))

    table = []
    for run in runs:
        name = run["run_name"]
        acc = load_json(Path(run["accuracy_curve_path"]))
        ent = load_json(Path(run["entropy_curve_path"]))

        ax1.plot([x["ei_step"] for x in acc], [x["validation_answer_accuracy"] for x in acc], marker="o", label=name)
        ax2.plot([x["ei_step"] for x in ent], [x["response_entropy"] for x in ent], marker="s", label=name)

        table.append(
            {
                "run_name": name,
                "rollout_g": run["rollout_g"],
                "sft_epochs": run["sft_epochs"],
                "db_size": run["db_size"],
                "final_validation_answer_accuracy": run["final_validation_answer_accuracy"],
            }
        )

    ax1.set_title("EI Validation Accuracy Curves")
    ax1.set_xlabel("EI step")
    ax1.set_ylabel("Validation answer accuracy")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_title("EI Response Entropy Curves")
    ax2.set_xlabel("EI step")
    ax2.set_ylabel("Response entropy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig1.tight_layout()
    fig2.tight_layout()

    acc_png = base / "ei_validation_accuracy_curves.png"
    ent_png = base / "ei_entropy_curves.png"
    fig1.savefig(acc_png, dpi=180)
    fig2.savefig(ent_png, dpi=180)

    with (base / "ei_results_table.json").open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    print(f"Saved: {acc_png}")
    print(f"Saved: {ent_png}")


if __name__ == "__main__":
    main()
