"""
Visualization utilities for NEAT training results.
Uses matplotlib only — no graphviz dependency.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; safe for saving without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_fitness_history(stats, filename: str = "fitness_history.png") -> None:
    """
    Plot per-generation best fitness, mean fitness, and mean±std band.

    Args:
        stats: neat.StatisticsReporter instance (after pop.run() completes)
        filename: output PNG path
    """
    generations = list(range(len(stats.most_fit_genomes)))
    max_fitness = [g.fitness for g in stats.most_fit_genomes]
    mean_fitness = np.array(stats.get_fitness_mean())
    stdev_fitness = np.array(stats.get_fitness_stdev())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(generations, max_fitness, "b-", label="Best fitness", linewidth=2)
    ax.plot(generations, mean_fitness, "g-", label="Mean fitness", linewidth=1.5)
    ax.fill_between(
        generations,
        mean_fitness - stdev_fitness,
        mean_fitness + stdev_fitness,
        alpha=0.3,
        color="green",
        label="Mean ± 1 std",
    )
    ax.axhline(y=200, color="r", linestyle="--", linewidth=1, label="Solved threshold (200)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("NEAT LunarLander-v3 Training Progress")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Fitness history saved to {filename}")


def visualize_network(config, genome, filename: str = "winner_network.png") -> None:
    """
    Draw the NEAT network topology using matplotlib circles and lines.

    Layout: inputs (left column) → hidden (middle) → outputs (right column).
    Connection color: blue = positive weight, red = negative weight.
    Connection width: proportional to abs(weight).

    Args:
        config: neat.Config object
        genome: winning genome (neat.DefaultGenome)
        filename: output PNG path
    """
    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)
    hidden_keys = [k for k in genome.nodes if k not in output_keys]

    def column_y_positions(keys, x: float) -> dict:
        n = len(keys)
        if n == 0:
            return {}
        ys = np.linspace(0.1, 0.9, n) if n > 1 else [0.5]
        return {k: (x, float(y)) for k, y in zip(sorted(keys), ys)}

    node_pos = {}
    node_pos.update(column_y_positions(input_keys, 0.1))
    node_pos.update(column_y_positions(output_keys, 0.9))
    if hidden_keys:
        node_pos.update(column_y_positions(hidden_keys, 0.5))

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fitness_label = f"{genome.fitness:.1f}" if genome.fitness is not None else "N/A"
    ax.set_title(f"NEAT Network Topology  (fitness = {fitness_label})", fontsize=14)

    # Draw connections
    weights = [abs(c.weight) for c in genome.connections.values() if c.enabled]
    max_weight = max(weights) if weights else 1.0

    for cg in genome.connections.values():
        if not cg.enabled:
            continue
        src, dst = cg.key
        if src not in node_pos or dst not in node_pos:
            continue
        x0, y0 = node_pos[src]
        x1, y1 = node_pos[dst]
        color = "steelblue" if cg.weight >= 0 else "tomato"
        lw = 0.5 + 3.0 * abs(cg.weight) / (max_weight + 1e-8)
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.6, zorder=1)

    # Node colors by role
    color_map = {}
    for k in input_keys:
        color_map[k] = "lightgreen"
    for k in output_keys:
        color_map[k] = "lightsalmon"
    for k in hidden_keys:
        color_map[k] = "lightblue"

    input_labels = ["x_pos", "y_pos", "x_vel", "y_vel", "angle", "ang_vel", "leg_L", "leg_R"]
    output_labels = ["nothing", "left_eng", "main_eng", "right_eng"]

    for key, (x, y) in node_pos.items():
        circle = plt.Circle(
            (x, y), 0.03,
            color=color_map.get(key, "lightblue"),
            ec="black", lw=1.0, zorder=2,
        )
        ax.add_patch(circle)

        if key in input_keys:
            idx = input_keys.index(key)
            label = input_labels[idx] if idx < len(input_labels) else str(key)
            ax.text(x - 0.055, y, label, ha="right", va="center", fontsize=7)
        elif key in output_keys:
            idx = output_keys.index(key)
            label = output_labels[idx] if idx < len(output_labels) else str(key)
            ax.text(x + 0.055, y, label, ha="left", va="center", fontsize=7)
        else:
            ax.text(x, y + 0.045, str(key), ha="center", va="bottom", fontsize=7)

    legend_patches = [
        mpatches.Patch(color="lightgreen", label="Input nodes"),
        mpatches.Patch(color="lightsalmon", label="Output nodes"),
        mpatches.Patch(color="lightblue", label="Hidden nodes"),
    ]
    ax.legend(handles=legend_patches, loc="upper center", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Network visualization saved to {filename}")
