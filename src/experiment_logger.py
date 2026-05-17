import csv
import json
import os


def get_experiment_dir(base_results_dir, sim_config_full):
    """ Create a results folder for the current fitness strategy. """
    strategy_name = sim_config_full["simulation"].get("fitness_strategy", "default")
    experiment_dir = os.path.join(base_results_dir, strategy_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_config_used(sim_config_full, filename):
    """ Save the configuration used in the experiment to a JSON file. """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(sim_config_full, file, indent=2)


def save_training_stats(stats, filename):
    """
    Save training statistics collected by NEAT to a CSV file.

    The saved file contains one row per generation with:
    generation number, best fitness, mean fitness and standard deviation.
    This is useful for comparing different fitness strategies later.
    """
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