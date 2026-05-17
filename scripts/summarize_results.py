import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
SOLVED_THRESHOLD = 200.0


def read_training_stats(csv_path):
    """
    Read training statistics saved for one experiment.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "generation": int(float(row["generation"])),
                    "best_fitness": float(row["best_fitness"]),
                    "mean_fitness": float(row["mean_fitness"]),
                    "std_fitness": float(row["std_fitness"]),
                }
            )
    return rows


def summarize_experiment(strategy_dir):
    """
    Compute summary statistics for one fitness strategy.
    """
    stats_path = strategy_dir / "training_stats.csv"
    config_path = strategy_dir / "config_used.json"

    if not stats_path.exists() or not config_path.exists():
        return None

    stats = read_training_stats(stats_path)

    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)

    strategy = config["simulation"]["fitness_strategy"]
    best_row = max(stats, key=lambda row: row["best_fitness"])
    final_row = stats[-1]

    solved_generations = [
        row["generation"]
        for row in stats
        if row["best_fitness"] >= SOLVED_THRESHOLD
    ]

    first_solved_generation = (
        min(solved_generations) if solved_generations else ""
    )

    return {
        "strategy": strategy,
        "num_generations": len(stats),
        "best_fitness": best_row["best_fitness"],
        "best_generation": best_row["generation"],
        "final_best_fitness": final_row["best_fitness"],
        "final_mean_fitness": final_row["mean_fitness"],
        "final_std_fitness": final_row["std_fitness"],
        "first_solved_generation": first_solved_generation,
    }


def save_summary(summary_rows):
    """
    Save one CSV file comparing all experiment folders.
    """
    output_path = RESULTS_DIR / "summary.csv"

    fieldnames = [
        "strategy",
        "num_generations",
        "best_fitness",
        "best_generation",
        "final_best_fitness",
        "final_mean_fitness",
        "final_std_fitness",
        "first_solved_generation",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to {output_path}")


def save_comparison_plot():
    """
    Save one plot comparing best fitness curves for all strategies.
    """
    plt.figure(figsize=(12, 6))

    for strategy_dir in sorted(RESULTS_DIR.iterdir()):
        if not strategy_dir.is_dir():
            continue

        stats_path = strategy_dir / "training_stats.csv"
        config_path = strategy_dir / "config_used.json"

        if not stats_path.exists() or not config_path.exists():
            continue

        stats = read_training_stats(stats_path)

        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)

        strategy = config["simulation"]["fitness_strategy"]
        generations = [row["generation"] for row in stats]
        best_values = [row["best_fitness"] for row in stats]

        plt.plot(generations, best_values, label=strategy)

    plt.axhline(
        y=SOLVED_THRESHOLD,
        linestyle="--",
        linewidth=1,
        label="Solved threshold (200)",
    )
    plt.xlabel("Generation")
    plt.ylabel("Best strategy-specific fitness")
    plt.title("Best strategy-specific fitness comparison across strategies")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = RESULTS_DIR / "best_fitness_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved comparison plot to {output_path}")


def main():
    summary_rows = []

    for strategy_dir in sorted(RESULTS_DIR.iterdir()):
        if not strategy_dir.is_dir():
            continue

        summary = summarize_experiment(strategy_dir)
        if summary is not None:
            summary_rows.append(summary)

    save_summary(summary_rows)
    save_comparison_plot()


if __name__ == "__main__":
    main()