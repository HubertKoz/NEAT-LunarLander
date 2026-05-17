import csv
import json
import os


def get_experiment_dir(base_results_dir, sim_config_full):
    strategy_name = sim_config_full["simulation"].get("fitness_strategy", "default")
    experiment_dir = os.path.join(base_results_dir, strategy_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_config_used(sim_config_full, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(sim_config_full, file, indent=2)


def save_training_stats(stats, filename):
    max_fitness = [genome.fitness for genome in stats.most_fit_genomes]
    mean_fitness = stats.get_fitness_mean()
    std_fitness = stats.get_fitness_stdev()

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["generation", "best_fitness", "mean_fitness", "std_fitness"])

        for generation, best, mean, std in zip(
            range(len(max_fitness)),
            max_fitness,
            mean_fitness,
            std_fitness,
        ):
            writer.writerow([generation, best, mean, std])