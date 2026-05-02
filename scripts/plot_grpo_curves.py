"""
Plot GRPO training curves:
  - train reward mean over steps
  - validation answer accuracy over steps
  - response entropy over steps (if logged)
  - example rollouts printed to stdout + saved as .txt

Usage:
  python scripts/plot_grpo_curves.py --output-dir outputs/grpo_math
"""
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/grpo_math")
    args = p.parse_args()

    out = Path(args.output_dir)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        has_mpl = True
    except ImportError:
        has_mpl = False

    # -----------------------------------------------------------------------
    # Train reward curve
    # -----------------------------------------------------------------------
    reward_path = out / "train_reward_curve.json"
    val_path = out / "val_curve.json"
    rollout_path = out / "example_rollouts.json"

    if reward_path.exists() and val_path.exists():
        reward_data = load_json(reward_path)
        val_data = load_json(val_path)

        steps_r = [d["step"] for d in reward_data]
        rewards = [d["reward_mean"] for d in reward_data]
        entropies = [d.get("response_entropy") for d in reward_data]
        has_entropy = any(e is not None for e in entropies)
        if has_entropy:
            entropies_clean = [(s, e) for s, e in zip(steps_r, entropies) if e is not None]

        steps_v = [d["step"] for d in val_data]
        val_acc = [d["answer_accuracy"] for d in val_data]

        print("\n=== Validation Answer Accuracy ===")
        for s, a in zip(steps_v, val_acc):
            print(f"  step {s:4d}: {a:.4f}")

        print("\n=== Train Reward Mean (sample) ===")
        for d in reward_data[::max(1, len(reward_data) // 20)]:
            print(f"  step {d['step']:4d}: reward={d['reward_mean']:.4f}")

        if has_mpl:
            n_plots = 2 + (1 if has_entropy else 0)
            fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

            axes[0].plot(steps_r, rewards, color="tab:blue")
            axes[0].set_title("Train Reward Mean (per step)")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Reward Mean")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(steps_v, val_acc, marker="o", color="tab:orange")
            axes[1].set_title("Validation Answer Accuracy")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.3)

            if has_entropy:
                es, ev = zip(*entropies_clean)
                axes[2].plot(list(es), list(ev), color="tab:green")
                axes[2].set_title("Response Token Entropy")
                axes[2].set_xlabel("Step")
                axes[2].set_ylabel("Entropy (nats)")
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(out / "grpo_curves.png", dpi=150)
            print(f"\nSaved: {out / 'grpo_curves.png'}")
        else:
            print("\nmatplotlib not available — skipping PNG plot.")
    else:
        print("Training output files not found. Run grpo_train.py first.")

    # -----------------------------------------------------------------------
    # Example rollouts
    # -----------------------------------------------------------------------
    if rollout_path.exists():
        rollout_data = load_json(rollout_path)
        lines = []
        for entry in rollout_data:
            lines.append(f"\n{'='*60}")
            lines.append(f"  Step {entry['step']}")
            lines.append(f"{'='*60}")
            for i, ex in enumerate(entry.get("examples", []), 1):
                lines.append(f"\n[Example {i}]")
                lines.append(f"PROMPT (truncated):\n{ex['prompt']}")
                lines.append(f"\nRESPONSE (truncated):\n{ex['response']}")
                lines.append("")

        print("\n=== Example Rollouts ===")
        print("\n".join(lines))

        rollout_txt = out / "example_rollouts.txt"
        rollout_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSaved: {rollout_txt}")


if __name__ == "__main__":
    main()
