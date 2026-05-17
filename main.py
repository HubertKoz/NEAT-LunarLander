"""
NEAT LunarLander-v3 — main entry point.

Architecture notes:
- ParallelEvaluator is created ONCE before pop.run() to avoid spawning/destroying
  a process pool each generation (very expensive on Windows with 'spawn' start method).

- CURRENT_GENERATION is a module-level int incremented once per eval_genomes call so
  the worker callable (which receives only genome + config) still has access to the
  generation index for dynamic seeding.

- On Windows the 'spawn' start method requires:
    1. if __name__ == "__main__": guard around main()
    2. multiprocessing.freeze_support() call
    3. The worker function must be picklable — use a top-level class with __call__
       (_Worker) rather than a lambda or a nested closure.
"""

import json
import glob
import multiprocessing
import os

import neat
import neat.nn

from src.simulation import LunarSimulation
from src.visualization import plot_fitness_history, visualize_network
from src.experiment_logger import get_experiment_dir, save_config_used, save_training_stats


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Module-level generation counter (updated once per eval_genomes call)
CURRENT_GENERATION = 0


def load_configs():
    """Load simulation_config.json and neat_config.txt."""
    base = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base, "config")

    with open(os.path.join(config_dir, "simulation_config.json")) as f:
        sim_config_full = json.load(f)

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(config_dir, "neat_config.txt"),
    )
    return sim_config_full, neat_config


def find_latest_checkpoint(results_dir):
    """Return the checkpoint path with the highest generation number, or None."""
    prefix = os.path.join(results_dir, "neat-checkpoint-")
    files = glob.glob(f"{prefix}*")
    if not files:
        return None

    def parse_gen(fname):
        try:
            return int(os.path.basename(fname).replace("neat-checkpoint-", ""))
        except ValueError:
            return -1

    return max(files, key=parse_gen)


class _Worker:
    """
    Picklable callable for ParallelEvaluator.

    A top-level class with __call__ is used instead of a nested closure
    because Windows 'spawn' start-method requires all objects sent to
    worker processes to be picklable — closures are not.
    """

    def __init__(self, simulator: LunarSimulation):
        self.simulator = simulator

    def __call__(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        return self.simulator.run_agent(net, render_mode=None,
                                        generation_idx=CURRENT_GENERATION)


def make_eval_genomes(pe, sim_config_full, simulator, neat_config):
    """Return the eval_genomes function passed to pop.run()."""
    render_flag = sim_config_full["neat_runtime"].get("render_best_after_generation", False)

    def eval_genomes(genomes, config):
        global CURRENT_GENERATION

        # Parallel fitness evaluation (blocks until all workers finish)
        pe.evaluate(genomes, config)

        # Optionally render the best genome of this generation
        if render_flag:
            best_genome = max(genomes, key=lambda gv: gv[1].fitness)[1]
            net = neat.nn.FeedForwardNetwork.create(best_genome, config)
            print(f"\n--- Rendering best genome (gen {CURRENT_GENERATION}, "
                  f"fitness={best_genome.fitness:.2f}) ---")
            simulator.run_agent(net, render_mode="human",
                                generation_idx=CURRENT_GENERATION)

        CURRENT_GENERATION += 1

    return eval_genomes


def main():
    global CURRENT_GENERATION

    sim_config_full, neat_config = load_configs()
    sim_cfg = sim_config_full["simulation"]
    runtime_cfg = sim_config_full["neat_runtime"]
    
    experiment_dir = get_experiment_dir(RESULTS_DIR, sim_config_full)
    print(f"Experiment outputs will be saved to: {experiment_dir}")

    simulator = LunarSimulation(sim_cfg)
    worker_fn = _Worker(simulator)

    # One process pool for the entire training run
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    pe = neat.ParallelEvaluator(num_workers=num_workers, eval_function=worker_fn)

    # Build or restore population
    if runtime_cfg.get("resume_from_latest_checkpoint", False):
        checkpoint_file = find_latest_checkpoint(experiment_dir)
        if checkpoint_file:
            print(f"Restoring from checkpoint: {checkpoint_file}")
            pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
            CURRENT_GENERATION = pop.generation
        else:
            print("No checkpoint found — starting fresh.")
            pop = neat.Population(neat_config)
            CURRENT_GENERATION = 0
    else:
        pop = neat.Population(neat_config)
        CURRENT_GENERATION = 0

    stats = neat.StatisticsReporter()
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(stats)
    pop.add_reporter(
        neat.Checkpointer(
            generation_interval=runtime_cfg.get("checkpoint_generation_interval", 10),
            filename_prefix=os.path.join(experiment_dir, "neat-checkpoint-"),
        )
    )

    eval_genomes = make_eval_genomes(pe, sim_config_full, simulator, neat_config)

    try:
        winner = pop.run(eval_genomes, n=300)
    finally:
        pe.close()

    print(f"\nBest genome fitness: {winner.fitness:.2f}")

    # Post-training visualizations
    save_config_used(sim_config_full, filename=os.path.join(experiment_dir, "config_used.json"))

    save_training_stats(stats, filename=os.path.join(experiment_dir, "training_stats.csv"))

    plot_fitness_history(stats, filename=os.path.join(experiment_dir, "fitness_history.png"))

    visualize_network(neat_config, winner, filename=os.path.join(experiment_dir, "winner_network.png"))

    # Render the winner
    print("\n--- Rendering winner genome ---")
    net = neat.nn.FeedForwardNetwork.create(winner, neat_config)
    simulator.run_agent(net, render_mode="human", generation_idx=0)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required on Windows with 'spawn' start method
    main()
