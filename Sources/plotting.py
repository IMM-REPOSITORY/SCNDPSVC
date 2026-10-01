
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache_' + os.environ['USER']

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import math


NODE_SIZE = 800
EDGE_RAD = 0.12
EDGE_AVOID_RAD = 0.36
EDGE_NODE_CLEARANCE = 0.7
ARROW_SIZE = 14
ARROW_MARGIN = 24


def point_to_segment_distance(point, start, end):
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def edge_connectionstyle(edge, pos):
    u, v = edge
    start = pos[u]
    end = pos[v]
    closest_node = None
    closest_distance = None

    for node, point in pos.items():
        if node in edge:
            continue
        distance = point_to_segment_distance(point, start, end)
        if closest_distance is None or distance < closest_distance:
            closest_node = node
            closest_distance = distance

    if closest_distance is None or closest_distance >= EDGE_NODE_CLEARANCE:
        return f"arc3,rad={EDGE_RAD}"

    x1, y1 = start
    x2, y2 = end
    px, py = pos[closest_node]
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    rad = -EDGE_AVOID_RAD if cross >= 0 else EDGE_AVOID_RAD
    return f"arc3,rad={rad}"


def draw_directed_edges(G, pos, edges, ax, color, width=1.0):
    for edge in edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[edge],
            edge_color=color,
            width=width,
            arrowstyle="->",
            arrowsize=ARROW_SIZE,
            connectionstyle=edge_connectionstyle(edge, pos),
            min_source_margin=ARROW_MARGIN,
            min_target_margin=ARROW_MARGIN,
            ax=ax,
        )


def draw_clusters(ax, clusters, pos):
    for cluster_id, vertices in clusters.items():
        xs = [pos[v][0] for v in vertices]
        ys = [pos[v][1] for v in vertices]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        rx = (max(xs) - min(xs)) / 2 + 0.4
        ry = (max(ys) - min(ys)) / 2 + 0.4

        rx = max(rx, 0.5)
        ry = max(ry, 0.5)

        ellipse = patches.Ellipse(
            (cx, cy),
            width=1.5 * rx,
            height=1.5 * ry,
            edgecolor="red",
            facecolor="red",
            alpha=0.1,
            linestyle="none",
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(ellipse)

        ax.text(
            cx,
            cy + ry + 0.02,
            rf"$\mathfrak{{M}}_{{{cluster_id}}}$",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )


def get_legend_elements():
    return [
        Line2D([0], [0], color="darkblue", lw=2.5, label="Включено в решение"),
        Line2D([0], [0], color="lightgray", lw=1, linestyle="dashed", label="Не включено"),
    ]


def draw_topology(data, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    nx.draw(data.H, data.h_pos, with_labels=True, font_size=9, node_color="w", edgecolors="blue", ax=ax)
    return ax


def draw_input_graph(data, ax=None, pos=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    if pos is None:
        pos = data.pos

    draw_directed_edges(data.G, pos, list(data.G.edges()), ax, color="gray", width=1.0)
    nx.draw_networkx_nodes(data.G, pos, node_color="lightblue", node_size=NODE_SIZE, ax=ax)
    nx.draw_networkx_labels(data.G, pos, font_size=12, font_weight="bold", ax=ax)

    edge_labels = {(u, v): f"c={d['c']}\np={d['p']}" for u, v, d in data.G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        data.G, pos, edge_labels=edge_labels,
        font_size=7, label_pos=0.5,
        bbox=dict(alpha=0.6, facecolor="white", edgecolor="none"), ax=ax,
    )

    draw_clusters(ax, data.clusters, pos)
    ax.axis("off")
    return ax

import numpy as np
from matplotlib.path import Path as MplPath

def _bezier_point(p0, p1, p2, t):
    return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2


def draw_directed_edges_with_labels(G, pos, edges, ax, color, width=1.0):
    for edge in edges:
        # Рисуем стрелку
        nx.draw_networkx_edges(
            G, pos, edgelist=[edge], edge_color=color, width=width,
            arrowstyle="->", arrowsize=ARROW_SIZE,
            connectionstyle=edge_connectionstyle(edge, pos),
            min_source_margin=ARROW_MARGIN, min_target_margin=ARROW_MARGIN,
            ax=ax,
        )

        # --- Вычисляем координаты изогнутой дуги ---
        u, v = edge
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        cross = (x2 - x1) * (y2 - y1)
        style = edge_connectionstyle(edge, pos)
        import re
        m = re.search(r"rad=(-?\d+\.?\d*)", style)
        rad = float(m.group(1)) if m else EDGE_RAD

        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        px, py = -dy / length, dx / length
        cx = mx + rad * px
        cy = my + rad * py

        t = 0.5
        p0 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p1 = np.array([cx, cy])
        point = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2

        label = f"c={G.edges[edge]['c']}\np={G.edges[edge]['p']}"
        ax.text(point[0], point[1], label, fontsize=6,
                ha='center', va='center',
                bbox=dict(alpha=0.5, facecolor='white', edgecolor='none', pad=0.5))

def draw_solution(data, x_solution, ax=None, pos=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    if pos is None:
        pos = data.pos

    inactive_edges = [e for e in data.G.edges() if x_solution[e] == 0]
    nx.draw_networkx_edges(
        data.G, pos, edgelist=inactive_edges,
        edge_color="lightgray", style="dashed", arrows=False, ax=ax,
    )

    active_edges = [e for e in data.G.edges() if x_solution[e] == 1]
    draw_directed_edges_with_labels(data.G, pos, active_edges, ax,
                                    color="darkblue", width=2.5)

    nx.draw_networkx_nodes(data.G, pos, node_color="lightblue", node_size=NODE_SIZE, ax=ax)
    nx.draw_networkx_labels(data.G, pos, font_size=12, font_weight="bold", ax=ax)

    draw_clusters(ax, data.clusters, pos)
    ax.legend(handles=get_legend_elements(), loc="lower right")
    ax.axis("off")
    return ax