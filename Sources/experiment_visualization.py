"""Сохранение визуализаций экспериментов на диск."""

from pathlib import Path
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache_' + os.environ['USER']

import matplotlib.pyplot as plt

from plotting import draw_input_graph, draw_solution, draw_topology


def safe_stem(path):
    return Path(path).stem.replace(".", "_")


def save_figure(ax, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"Saved: {path}")


def instance_output_dir(output_dir, instance_file):
    return Path(output_dir) / safe_stem(instance_file)


def save_base_visualizations(data, output_dir):
    topology_ax = draw_topology(data)
    save_figure(topology_ax, output_dir / "topology.png")

    input_ax = draw_input_graph(data)
    save_figure(input_ax, output_dir / "input_graph.png")


def save_experiment_solution(data, result, experiment, output_dir, pos=None):
    solution_ax = draw_solution(data, result.x_solution, pos=pos)
    save_figure(solution_ax, output_dir / f"solution_{experiment}.png")
