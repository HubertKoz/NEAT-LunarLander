# Neuroevolution of Neural Networks in the Lunar Lander Environment (NEAT)

**Polish version:** [README.md](README.md)  
**Detailed experiment analysis:** [EXPERIMENTS.md](EXPERIMENTS.md)

## Project Description and Goal

This research project focuses on the analysis of neuroevolutionary processes using the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm. The main goal is to evolve an autonomous agent, a lunar lander, that learns how to land safely and precisely on uneven terrain in the **Lunar Lander** simulation provided by the *Gymnasium* library.

Unlike classical reinforcement learning methods, where the neural network architecture is fixed, NEAT allows both synaptic weights and the **structure (topology) of the network** to evolve at the same time. The process starts from minimal forms and gradually increases complexity through natural selection.

---

## Key Assumptions and Research Possibilities

### 1. Fitness Function Engineering

The core of the project is the study of how evaluation criteria affect agent behavior. The architecture allows reward strategies to be changed dynamically. The project implements five fitness strategies:

| Strategy | Identifier | Logic |
|---|---|---|
| **Default** | `"default"` | Native reward system of the Gymnasium environment |
| **Time Pressure** | `"penalty_time"` | Constant penalty of `−0.2` for each simulation step, encouraging faster episode completion |
| **Angle Stabilization** | `"penalty_angle"` | Penalty for lander tilt at the moment of ground contact |
| **Landing Quality** | `"landing_quality"` | Penalty for distance from the center, velocity and tilt, with an additional bonus for stable two-leg contact |
| **Centered Landing** | `"centered_landing"` | Stronger penalty for moving away from the center of the landing pad |

Changing the strategy does not require modifying the code — it is enough to edit the `fitness_strategy` field in `config/simulation_config.json`.

### 2. Control of Environmental Conditions: Generalization vs. Overfitting

To make agent evaluation fair and meaningful, the system allows each neural network to be tested over several independent runs within one generation. Depending on the experiment configuration, it is possible to study:

- **Memorization ability (fixed seeds):** Agents are tested on the same predefined terrain layouts, which allows their performance in a known environment to be compared precisely.
- **Generalization ability (variable seeds):** The terrain changes dynamically from generation to generation, forcing the population to develop more universal behaviors and preventing it from “memorizing the map”.

The `use_fixed_seeds` parameter in the configuration file switches between these two modes.

### 3. Scalability and Progress Analysis

The evolution process supports full parallel processing through `ParallelEvaluator`, so hundreds of individuals can be evaluated simultaneously on all available CPU cores. For demonstration and analysis purposes, the system allows live preview of the “best individual” from a given generation. In addition, the learning process is protected by automatic checkpoints, which makes it possible to run long-term experiments, stop simulations and resume them later without losing the evolved gene pool.

---

## Project Architecture

```text
project/
├── config/
│   ├── neat_config.txt          # NEAT algorithm parameters: population size, mutations, speciation
│   └── simulation_config.json   # experiment flags: strategy, seeds, rendering, checkpoints
├── src/
│   ├── fitness_functions.py     # 5 fitness strategies + FITNESS_STRATEGIES dictionary
│   ├── experiment_logger.py     # saving experiment configuration and statistics
│   ├── simulation.py            # LunarSimulation class — Gymnasium episode loop
│   └── visualization.py         # training progress and network topology plots
├── scripts/
│   └── summarize_results.py     # generates summary.csv and the comparison plot
├── results/
│   └── <fitness_strategy>/      # results of separate experiments
├── EXPERIMENTS.md               # detailed analysis of results
├── main.py                      # orchestration: NEAT loop, ParallelEvaluator, checkpoints
└── requirements.txt
```

### Data Flow in One Generation

```text
Population (150 genomes)
    │
    ▼  [parallel execution on N CPU cores]
ParallelEvaluator
    │   for each genome:
    │     FeedForwardNetwork.create(genome, config)
    │     LunarSimulation.run_agent(net)  ← 3 episodes × max 500 steps
    │       env.reset(seed) → loop: net.activate(obs) → argmax → env.step()
    │       fitness += strategy_func(reward, info, step, obs)
    │     return mean(fitness over 3 runs)
    │
    ▼
Selection + Speciation + Reproduction (NEAT)
    │
    ▼  [every 10 generations]
Checkpoint saved → neat-checkpoint-N
```

---

## State and Action Space

The agent observes 8 continuous values describing the state of the lander:

| Index | Observation | Description |
|---|---|---|
| 0 | `x_pos` | Horizontal position |
| 1 | `y_pos` | Vertical position |
| 2 | `x_vel` | Horizontal velocity |
| 3 | `y_vel` | Vertical velocity |
| 4 | `angle` | Tilt angle in radians |
| 5 | `ang_vel` | Angular velocity |
| 6 | `leg_L` | Left leg ground contact, 0/1 |
| 7 | `leg_R` | Right leg ground contact, 0/1 |

Based on these observations, the network selects one of 4 discrete actions:

| Output | Action |
|---|---|
| 0 | Do nothing, free flight |
| 1 | Fire left side engine |
| 2 | Fire main engine, downward-to-upward thrust |
| 3 | Fire right side engine |

Action selection: `action = argmax(net.activate(observation))`.

---

## NEAT Configuration (`config/neat_config.txt`)

Key algorithm parameters:

| Parameter | Value | Justification |
|---|---|---|
| `pop_size` | 150 | Sufficient diversity without excessive computational cost |
| `fitness_threshold` | 250.0 | Reference threshold above the environment’s “solved” threshold. With `no_fitness_termination = True`, it does not stop the experiment early |
| `num_hidden` | 0 | Starts with a minimal network — NEAT adds nodes on its own |
| `initial_connection` | `full_direct` | Full direct connection from inputs to outputs at the start |
| `activation_default` | `tanh` | Fixed to tanh: stable values, output in \[−1, 1\] |
| `elitism` | 2 | Top 2 genomes of each species survive without mutation |
| `max_stagnation` | 20 | A species has 20 generations to improve before being removed |
| `compatibility_threshold` | 3.0 | Genomic distance threshold used for species formation |
| `no_fitness_termination` | `True` | Forces all experiments to run for the full 300 generations, even if `fitness_threshold` is reached |

---

## Experiment Configuration File (`config/simulation_config.json`)

```json
{
  "simulation": {
    "max_steps_per_episode": 500,
    "num_eval_runs": 3,
    "use_fixed_seeds": true,
    "fixed_seeds": [42, 123, 999],
    "fitness_strategy": "default"
  },
  "neat_runtime": {
    "checkpoint_generation_interval": 10,
    "resume_from_latest_checkpoint": false,
    "render_best_after_generation": true
  }
}
```

| Parameter | Description |
|---|---|
| `fitness_strategy` | `"default"` / `"penalty_time"` / `"penalty_angle"` / `"landing_quality"` / `"centered_landing"` |
| `use_fixed_seeds` | `true` → fixed seeds for comparability; `false` → variable seeds for generalization |
| `fixed_seeds` | List of environment seeds — one seed per evaluation run |
| `num_eval_runs` | Number of evaluation runs per genome; fitness is averaged |
| `resume_from_latest_checkpoint` | `true` → resume from the latest `neat-checkpoint-N` |
| `render_best_after_generation` | `true` → visualization window for the best agent after each generation |

---

## Installation and Running

### Requirements

```text
neat-python >= 2.0.0
gymnasium[box2d] >= 1.0.0
matplotlib >= 3.7.0
numpy >= 1.24.0
swig
```

### Installing Dependencies

```bash
pip install swig
pip install neat-python "gymnasium[box2d]"
```

> **Note for Python 3.13 / Windows:** If `gymnasium[box2d]` fails to install because compiled Box2D wheels are missing, install `swig` first, which is a tool for generating C bindings, and then try again. Alternatively, use Python 3.11, for which prebuilt binary wheels are available.

### Running the Project

```bash
python main.py
```

The program automatically:

1. Starts evolution from generation 0, or resumes from a checkpoint if `resume_from_latest_checkpoint: true`.
2. Displays statistics after each generation.
3. Renders the best agent of the generation if `render_best_after_generation: true`.
4. Saves a checkpoint every `checkpoint_generation_interval` generations.
5. After completion, saves results to `results/<fitness_strategy>/`, including `fitness_history.png`, `winner_network.png`, `training_stats.csv` and `config_used.json`.

### Resuming Interrupted Training

In `simulation_config.json`, set:

```json
"resume_from_latest_checkpoint": true
```

The program will find the `neat-checkpoint-N` file with the highest `N` and resume evolution from that checkpoint.

---

## Results and Visualizations

Results are saved in a folder corresponding to the selected fitness strategy:

```text
results/<fitness_strategy>/
```

After each training run, 4 files are generated:

- **fitness_history.png** — progress plot per generation: maximum fitness, mean fitness and standard deviation band. The horizontal dashed line indicates the environment’s “solved” threshold of 200 points.
- **winner_network.png** — topology of the winning genome’s network: input nodes are green, hidden nodes are blue, output nodes are salmon-colored; blue edges represent positive weights, red edges represent negative weights; edge thickness is proportional to `|weight|`.
- **training_stats.csv** — table containing `best_fitness`, `mean_fitness` and `std_fitness` for each generation.
- **config_used.json** — configuration used to generate the given experiment.

The following script is used for collective experiment comparison:

```bash
python scripts/summarize_results.py
```

It creates:

- **results/summary.csv** — summary of results,
- **results/best_fitness_comparison.png** — plot comparing the strategies.

A detailed analysis of the experiments is available in [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Future Work

The system was designed in a modular way, which opens the path for further research:

- **Recurrent networks:** Replacing `neat.nn.FeedForwardNetwork` with `neat.nn.RecurrentNetwork` and setting `feed_forward = False` in `neat_config.txt` would allow agents to have short-term memory without any changes to the rest of the architecture.
- **Further reward strategies:** The project was extended with the `landing_quality` and `centered_landing` strategies, but other reward shaping variants can still be tested, such as separate penalties for fuel consumption, vertical velocity at ground contact, or a bonus only for successful episode completion.
- **Other Gymnasium environments:** The `LunarSimulation` class can be replaced with an analogous class for any discrete-action environment — the only required change is the number of inputs and outputs in `neat_config.txt`.