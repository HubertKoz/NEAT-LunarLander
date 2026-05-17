"""
LunarSimulation: wraps gymnasium episode execution for NEAT genome evaluation.
"""
import numpy as np
import gymnasium as gym

from src.fitness_functions import FITNESS_STRATEGIES


class LunarSimulation:
    def __init__(self, sim_config: dict):
        """
        sim_config: the "simulation" sub-dict from simulation_config.json, e.g.:
            {
                "max_steps_per_episode": 500,
                "num_eval_runs": 3,
                "use_fixed_seeds": True,
                "fixed_seeds": [42, 123, 999],
                "fitness_strategy": "penalty_angle"
            }
        """
        self.max_steps = sim_config["max_steps_per_episode"]
        self.num_eval_runs = sim_config["num_eval_runs"]
        self.use_fixed_seeds = sim_config["use_fixed_seeds"]
        self.fixed_seeds = sim_config["fixed_seeds"]

        strategy_name = sim_config.get("fitness_strategy", "default")
        if strategy_name not in FITNESS_STRATEGIES:
            raise ValueError(
                f"Unknown fitness strategy '{strategy_name}'. "
                f"Valid options: {list(FITNESS_STRATEGIES.keys())}"
            )
        self.strategy_func = FITNESS_STRATEGIES[strategy_name]

    def run_agent(self, net, render_mode=None, generation_idx: int = 0) -> float:
        """
        Evaluate a NEAT feed-forward network over num_eval_runs episodes.

        The environment is created and closed inside this method so that
        render_mode can differ between headless training and visual rendering.

        Args:
            net: neat.nn.FeedForwardNetwork (created from genome + config)
            render_mode: None for headless, "human" for visual display
            generation_idx: current generation number, used for dynamic seeding

        Returns:
            Average fitness across all evaluation runs.
        """
        total_fitness = 0.0
        env = gym.make("LunarLander-v3", render_mode=render_mode)

        for run_idx in range(self.num_eval_runs):
            if self.use_fixed_seeds:
                seed = self.fixed_seeds[run_idx % len(self.fixed_seeds)]
            else:
                # Same seed for entire population in a generation, varies per run
                seed = generation_idx * 1000 + run_idx

            observation, _ = env.reset(seed=seed)
            episode_fitness = 0.0

            for step_count in range(self.max_steps):
                action = int(np.argmax(net.activate(observation)))
                observation, gym_reward, terminated, truncated, info = env.step(action)

                episode_fitness += self.strategy_func(
                    gym_reward, info, step_count, observation
                )

                if terminated or truncated:
                    break

            total_fitness += episode_fitness

        env.close()
        return total_fitness / self.num_eval_runs
