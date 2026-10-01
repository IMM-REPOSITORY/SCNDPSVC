import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import time
import random as std_random
import math

from scipy.spatial.distance import directed_hausdorff

from itertools import accumulate, islice, combinations

def draw_clusters(pos, cluster_type, clusters, ax, labels = None):
    if labels is None:
        labels = [str(i) for i in range(len(clusters))]
        
    for i, vertices in enumerate(clusters):
        xs = [pos[v][0] for v in vertices]
        ys = [pos[v][1] for v in vertices]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        rx = (max(xs) - min(xs)) / 2 + 0.2
        ry = (max(ys) - min(ys)) / 2 + 0.5

        rx = max(rx, 0.5)
        ry = max(ry, 0.5)

        ellipse = patches.Ellipse(
            (cx, cy), width=2.5*rx, height=1.5*ry,
            edgecolor='red', facecolor='red',
            alpha = 0.1,
            linestyle='none', linewidth=2, zorder=1
        )
        ax.add_patch(ellipse)
        ax.text(cx, cy + 0.25, rf"$\mathfrak{{M}}_{{{i}}}$", ha='center', va='center',
                fontsize=9, color='darkred', fontweight='bold', zorder=3)


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_G(G, clusters, new_pos, X, Y, cluster_type, name=None):
    size = (min(X, 16), min(Y, 16))
    fig, ax = plt.subplots(figsize=size)

    nx.draw_networkx_edges(G, pos=new_pos, ax=ax, edge_color="gray", alpha=0.5)

    nodes_in_clusters = set().union(*clusters)

    non_cluster_nodes = [node for node in G.nodes() if node not in nodes_in_clusters]
    if non_cluster_nodes:
        nx.draw_networkx_nodes(
            G,
            pos=new_pos,
            nodelist=non_cluster_nodes,
            node_size=1200,
            node_color="none",  # Без заливки
            edgecolors="black",  # Черная обводка
            linewidths=1.5,
            ax=ax,
        )

    pastel_colors = ["#FFB3B3", "#FFE0B2", "#C8E6C9", "#B2DFDB", "#B3E5FC"]

    for i, cluster in enumerate(clusters):
        light_color = pastel_colors[i % len(pastel_colors)]

        nx.draw_networkx_nodes(
            G,
            pos=new_pos,
            nodelist=cluster,
            node_size=1200,
            node_color=[
                light_color
            ] * len(cluster),  
            edgecolors="black", 
            linewidths=1.5,
            ax=ax,
        )

    nx.draw_networkx_labels(G, pos=new_pos, font_size=12, font_color="black", font_weight="bold", ax=ax)

    if cluster_type == "dense":
        draw_clusters(new_pos, cluster_type, clusters, ax)

    plt.axis("off")

    if name:
        plt.savefig(name, bbox_inches="tight")
        plt.close(fig)



def get_random_grid_graph(X, Y, C_min, C_max, C_d_min, C_d_max, p_base, eps, prob_type='random'):
    H = nx.Graph()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 1), (1, -1)]

    #for weights generation
    alpha = 1.0 
    beta = 1.5
    gamma = 5.0

    for i in range(1, X + 1):
        for j in range(1, Y + 1):
            v = (i, j)
            H.add_node(v)
            for (a, b) in directions:
                u = (i + a, j + b)
                if u in H.nodes:
                    H.add_edge(v, u)

    pos = {v: v for v in H.nodes}

    G = nx.DiGraph()
    new_pos = dict()
    for (v, u) in H.edges:
        new_v = v[0] + (v[1] - 1) * X
        new_u = u[0] + (u[1] - 1) * X
        if (v[0] == u[0] and v[1] != u[1]) or (v[1] == u[1] and v[0] != u[0]):
            weight = np.random.randint(C_min, C_max + 1)
        else:
            weight = np.random.randint(C_d_min, C_d_max + 1)

        G.add_edge(new_v, new_u, weight=weight)
        G.add_edge(new_u, new_v, weight=weight)
        new_pos[new_v] = v
        new_pos[new_u] = u

    c_max = max(G.edges[e]['weight'] for e in G.edges)
    for e in G.edges:
        if prob_type == 'random':
            G.edges[e]['prob'] = np.random.uniform(1e-7, 1e-1)
        elif prob_type == 'correlated':
            c = G.edges[e]['weight']
            f_e = c / c_max
            p_e = p_base + (1 - p_base - eps) * f_e
            G.edges[e]['prob'] = 1 - p_e

        G.edges[e]['b'] = 1

    node_pos = {val : key for key, val in new_pos.items()}

    return G, new_pos, node_pos


def generate_random_graph(X, Y, prob_type):
    C_min = 1
    C_max = 10
    C_d_min = 10
    C_d_max = 20
    p_base = 0.90
    eps = 1e-7

    G, new_pos, node_pos = get_random_grid_graph(X, Y, C_min, C_max, C_d_min, C_d_max,
                                       p_base, eps, prob_type=prob_type)
    s, t = min(G.nodes), max(G.nodes)
    return G, new_pos, node_pos, s, t


def check_clusters_path(G, cluster_i, cluster_j, remove_nodes):
    G_cluster = G
    main_cluster_i = cluster_i[0]
    for v in cluster_i[1:]:
        G_cluster = nx.contracted_nodes(G_cluster, main_cluster_i, v)

    main_cluster_j = cluster_j[0]
    for v in cluster_j[1:]:
        G_cluster = nx.contracted_nodes(G_cluster, main_cluster_j, v)

    sub_G_cluster = G_cluster.subgraph((set(G_cluster.nodes) - remove_nodes) |
                                       {main_cluster_i, main_cluster_j})
    is_path = nx.has_path(sub_G_cluster, main_cluster_i, main_cluster_j)
    return is_path


def get_clusters_constant(G, in_X, in_Y, new_pos, clusters_centers):
    s, t = min(G.nodes), max(G.nodes)
    chosen_positions = []
    for (x, y) in clusters_centers:
        chosen_positions.append([(x, y), (x + 1, y), (x, y + 1)])

    clusters = [[next(filter(lambda pair: pair[1] == pos, new_pos.items()), (None,))[0]
                 for pos in cluster_pos] for cluster_pos in chosen_positions]
    returned_clusters = {i + 1: clusters[i] for i, _ in enumerate(clusters)}

    return s, t, returned_clusters


def path_weight(G, path):
    path_weight = 0
    path_edges = list(zip(path[:-1], path[1:]))
    for e in path_edges:
        path_weight += G.edges[e]['weight']
    return path_weight

def directed_hausdorff_graph_distance(G, clusters, A, B, weight=None):
    max_min_dist = 0
    remove_nodes = {v for cluster in clusters.values() for v in cluster}
    
    for a in A:
        min_dist_to_B = float('inf')
        for b in B:
            try:
                subgraph = G.subgraph((set(G.nodes) - remove_nodes) | {a, b})
                # Находим длину кратчайшего пути от 'a' к 'b'
                dist = nx.shortest_path_length(subgraph, source=a, target=b, weight='b')
                if dist < min_dist_to_B:
                    min_dist_to_B = dist
            except nx.NetworkXNoPath:
                continue
        
        if min_dist_to_B == float('inf'):
            return float('inf')
            
        if min_dist_to_B > max_min_dist:
            max_min_dist = min_dist_to_B
            
    return max_min_dist


def compute_budget_hausdorff(G, H, clusters, budget_delta = 1.1):
    L1 = -1
    for (i,j) in H.edges:
        i_to_j = directed_hausdorff_graph_distance(G, clusters, clusters[i], clusters[j])
        L1 = max(L1, i_to_j)
    
    budget_primary = {(i,j) : L1 for (i,j) in H.edges}
    if max(budget_primary.values()) == float('inf'):
        return float('inf'), float('inf')

    budget_backup = {(i,j) : math.ceil(budget_delta * budget_primary[i,j]) for (i,j) in H.edges}

    return budget_primary, budget_backup

def compute_budget_simple(G, H, s, t, clusters, budget_delta = 1):
    remove_nodes = {v for cluster in clusters.values() for v in cluster}
    budget_primary = {}
    for i, j in H.edges():
        budget_primary[i, j] = float('inf')
        for v in clusters[i]:
            for u in clusters[j]:
                subgraph = G.subgraph((set(G.nodes) - remove_nodes) | {v, u})
                if not nx.has_path(subgraph, v, u):
                    continue
                # Кратчайший путь по бюджету b
                path = nx.shortest_path(subgraph, v, u, weight='b')
                # Суммируем b вдоль пути
                b_sum = sum(subgraph.edges[p1, p2]['b'] for p1, p2 in zip(path[:-1], path[1:]))
                budget_primary[i, j] = min(budget_primary[i, j], b_sum)
        if budget_primary[i, j] == float('inf'):
            return None, None

    max_budget = max(budget_primary.values())
    budget_primary = {(i,j) : max_budget for (i,j) in H.edges}

    budget_backup = {(i, j): math.ceil(budget_delta * budget_primary[i, j])
                     for (i, j) in budget_primary.keys()}

    return budget_primary, budget_backup

def remove_edges_in_clusters(G, clusters):
    edges_to_remove = list()
    for cluster in clusters:
        for v in cluster:
            for u in cluster:
                if v == u:
                    continue
                edges_to_remove.append((v, u))
    G.remove_edges_from(edges_to_remove)


def get_dense_clusters(G, X, Y, new_pos, num_clusters, start_node=0, end_node=None,
                         min_size=2, max_size=4, gap=1):
    if end_node is None:
        end_node = max(G.nodes)

    used = set([start_node, end_node])
    forbidden = set()

    for node in (start_node, end_node):
        x0, y0 = new_pos[node]
        for dx in range(-gap, gap+1):
            for dy in range(-gap, gap+1):
                forbidden.add((x0+dx, y0+dy))

    clusters = {}

    for idx in range(1, num_clusters + 1):
        desired_size = std_random.randint(min_size, max_size)

        candidates = []
        for v in G.nodes:
            if v in used:
                continue
            x, y = new_pos[v]
            if any((x+dx, y+dy) in forbidden for dx in [0,1] for dy in [0,1]):
                continue
            if any(node in used for node, pos in new_pos.items() if pos in [(x,y),(x+1,y),(x,y+1),(x+1,y+1)]):
                continue
            candidates.append(v)
        if not candidates:
            break

        base = std_random.choice(candidates)
        x0, y0 = new_pos[base]

        cluster = [base]
        used.add(base)

        offsets = [(0,1), (1,0), (1,1)]
        std_random.shuffle(offsets)

        for dx, dy in offsets:
            if len(cluster) >= desired_size:
                break
            target_pos = (x0 + dx, y0 + dy)
            for v, pos in new_pos.items():
                if pos == target_pos and v not in used:
                    cluster.append(v)
                    used.add(v)
                    break

        while len(cluster) < desired_size:
            possible = []
            for v, pos in new_pos.items():
                if v not in used and (pos[0] in (x0, x0+1)) and (pos[1] in (y0, y0+1)):
                    possible.append(v)
            if not possible:
                break
            v = std_random.choice(possible)
            cluster.append(v)
            used.add(v)

        clusters[idx] = cluster

        for v in cluster:
            x, y = new_pos[v]
            for dx in range(-gap, gap+1):
                for dy in range(-gap, gap+1):
                    forbidden.add((x+dx, y+dy))

    return clusters

def get_sparse_clusters(G, X, Y, new_pos, node_pos, num_clusters, start_node=1, end_node=None,
                        min_size=2, max_size=4, gap=1, cluster_radius = 2):
    if end_node is None:
        end_node = max(G.nodes)

    forbidden = set()
    forbidden_nodes = set()

    for node in (start_node, end_node):
        x0, y0 = new_pos[node]
        for dx in range(-gap, gap+1):
            for dy in range(-gap, gap+1):
                if 1 <= x0 + dx <= X and 1 <= y0 + dy <= Y: forbidden.add((x0+dx, y0+dy))

    forbidden.add(new_pos[start_node])
    forbidden.add(new_pos[end_node])

    clusters = {}

    for idx in range(1, num_clusters + 1):
        desired_size = std_random.randint(min_size, max_size)
        candidates = list(set(G.nodes) - set([node_pos[pos] for pos in forbidden]) - {start_node, end_node})

        first_node = std_random.choice(candidates)
        first_node_pos = new_pos[first_node]
        forbidden.update([first_node_pos])

        available_pos = []
        for dx in range(max(1, first_node_pos[0] - cluster_radius), min(X, first_node_pos[0] + cluster_radius)+1):
            for dy in range(max(1, first_node_pos[1] - cluster_radius), min(Y, first_node_pos[1] + cluster_radius)+1):
                if (dx, dy) not in forbidden:
                    available_pos.append((dx, dy))

        additional_node_positions = std_random.sample(available_pos, desired_size - 1) if desired_size >= 3 else [std_random.choice(available_pos)]
        additional_nodes = [node_pos[pos] for pos in additional_node_positions]
        clusters[idx] = [first_node] + additional_nodes

        forbidden.update(additional_node_positions)


    return clusters

def write_instance(filename_base, G, H, clusters, start_cluster, dest_cluster,
                   budget_primary, budget_backup,
                   capacities_primary, capacities_backup,
                   repairing_costs, repairing_budget, epsilon_primary, epsilon_backup):
    scndpsvc_filename = filename_base + '.scndpsvc'
    hubs = list(G.nodes - set(v for c in clusters.values() for v in c))
    adjacent_matrix = {(e1, e2): attr['weight'] for e1, e2, attr in G.edges(data=True)}
    prob_matrix = {(e1, e2): attr['prob'] for e1, e2, attr in G.edges(data=True)}
    budget_costs = {(e1,e2): attr['b'] for e1, e2, attr in G.edges(data=True)}
    ordering_graph = {v: list(H.successors(v)) for v in H.nodes() if H.out_degree(v) > 0}

    with open(scndpsvc_filename, 'w') as f:
        f.write(f'NAME : {scndpsvc_filename}\n')
        f.write(f'TYPE : SCNDPSVC\n')
        f.write(f'DIMENSION : {len([elem for key, value in clusters.items() for elem in value]) + len(hubs)}\n')
        f.write(f'CLUSTERS_NUMBER : {len(clusters)}\n')
        f.write(f'HUBS_NUMBER : {len(hubs)}\n')
        f.write(f'EDGE_WEIGHT_TYPE : EXPLICIT \n')
        f.write(f'EDGE_WEIGHT_FORMAT : FULL_MATRIX \n')

        f.write(f'EDGE_WEIGHT_SECTION \n')
        f.write(f'{adjacent_matrix}\n')

        f.write(f'EDGE_PROB_SECTION \n')
        f.write(f'{prob_matrix}\n')

        f.write(f'EDGE_CAPACITY_SECTION_PRIMARY\n')
        f.write(f'{capacities_primary}\n')

        f.write(f'EDGE_CAPACITY_SECTION_BACKUP\n')
        f.write(f'{capacities_backup}\n')

        f.write(f'EDGE_BUDGET_SECTION\n')
        f.write(f'{budget_costs}\n')

        f.write(f'EDGE_REPAIRING_SECTION\n')
        f.write(f'{repairing_costs}\n')

        f.write(f'CLUSTERS_SETS_SECTION\n')
        f.write(f'{clusters}\n')

        f.write(f'CLUSTERS_SETS_ORDERING\n')
        f.write(f'{ordering_graph}\n')

        f.write(f'HUBS_SECTION\n')
        f.write(f'{hubs}\n')

        f.write(f'START_GROUP_SECTION \n')
        f.write(f'{start_cluster}\n')
        f.write(f'DEST_GROUP_SECTION \n')
        f.write(f'{dest_cluster}\n')

        f.write(f'BUDGET_SECTION_PRIMARY\n')
        f.write(f'{budget_primary}\n')

        f.write(f'BUDGET_SECTION_BACKUP\n')
        f.write(f'{budget_backup}\n')

        f.write(f'REPAIRING_BUDGET\n')
        f.write(f'{repairing_budget}\n')

        f.write(f'EPSILON_SECTION_PRIMARY\n')
        f.write(f'{epsilon_primary}\n')

        f.write(f'EPSILON_SECTION_BACKUP\n')
        f.write(f'{epsilon_backup}\n')

        f.write(f'EOF\n')

def main():
    X = 10
    Y = 10

    plt.figure()
    H1 = nx.DiGraph()
    H1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    pos1 = {0: (0,0), 1: (10,0), 2: (20,0), 3: (30,0), 4: (40,0)}
    nx.draw(H1,pos1, with_labels=True, node_size=1200, font_size=20, node_color='w', edgecolors='blue')
    # plt.savefig('H1.png')
    topology_name = 'H1'

    plt.figure()
    H2 = nx.DiGraph()
    H2.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])
    pos2 = {0: (0,0), 1: (10,10), 2: (10,-10), 3: (20,10), 4: (30,0)}
    nx.draw(H2,pos2, with_labels=True, node_size=1200, font_size=20, node_color='w', edgecolors='blue')
    # plt.savefig('H2.png')
    topology_name = 'H2'

    plt.figure()
    H3 = nx.DiGraph()
    H3.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)])
    pos3 = {0: (0,0), 1: (10,10), 2: (10,0), 3: (10,-10), 4: (20,0)}
    nx.draw(H3,pos3, with_labels=True, node_size=1200, font_size=20, node_color='w', edgecolors='blue')
    # plt.savefig('H3.png')
    topology_name = 'H3'


    plt.figure()
    H4 = nx.DiGraph()
    H4.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    pos4 = {0: (0,0), 1: (10,10), 2: (10,-10), 3: (20,0), 4: (30,0)}
    nx.draw(H4,pos4, with_labels=True, node_size=1200, font_size=20, node_color='w', edgecolors='blue')
    # plt.savefig('H4.png')
    topology_name = 'H4'

    plt.figure()
    H5 = nx.DiGraph()
    H5.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 4)])
    pos5 = {0: (0,0), 1: (10,10), 2: (10,-10), 3: (20,0), 4: (40,0)}
    nx.draw(H5,pos5, with_labels=True, node_size=1200, font_size=20, node_color='w', edgecolors='blue')
    # plt.savefig('H5.png')
    topology_name = 'H5'

    plt.figure()
    H6 = nx.DiGraph()
    H6.add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,3),(3,4)])
    pos6 = {0: (0,0), 1: (15,10), 2: (15,-10), 3: (30,0), 4: (40,0)}
    nx.draw(H6,pos6, with_labels=True, node_size=1200, font_size=20, node_color='w', edgecolors='blue')
    # plt.savefig('H6.png')
    topology_name = 'H6'

    plt.figure()
    H7 = nx.DiGraph()
    H7.add_edges_from([(0,1),(0,2),(1,2),(1,3),(3,4),(2,4)])
    pos7 = {0: (0,0), 1: (15,10), 2: (15,-10), 3: (25,10), 4: (40,0)}
    nx.draw(H7,pos7, with_labels=True, node_size=1200, font_size=20, node_color='w', edgecolors='blue')
    # plt.savefig('H7.png')
    topology_name = 'H7'

    plt.figure()
    H8 = nx.DiGraph()
    H8.add_edges_from([(0,1),(0,2),(1,3),(2,3),(1,4),(2,4),(3,4)])
    pos8 = {0: (0,0), 1: (20,10), 2: (20,-10), 3: (20,0), 4: (40,0)}
    nx.draw(H8,pos8, with_labels=True, node_size=1200,  font_size=20, node_color='w', edgecolors='blue')
    # plt.savefig('H8.png')
    topology_name = 'H8'


    prob_types = ['correlated', 'random']
    cluster_types = ['dense', 'sparse']
    topologies = {1: H1, 2: H2, 3: H3, 4: H4, 5: H5, 6: H6, 7: H7, 8: H8}
    instance_numbers = [i for i in range(0, 20)]
    budget_delta = 1.1

    for prob_type in prob_types:
        for cluster_type in cluster_types:
            for topology_number, H in topologies.items():

                print(f'Proceed with {prob_type=}, {cluster_type=}, {topology_number=}')
                topology_name = f'H{topology_number}'

                instance_prefix = f'examples{X*Y}v/{prob_type}/{cluster_type}/{topology_name}_{prob_type}_{cluster_type}'

                for i in instance_numbers:
                    clusters = {}
                    G, new_pos, node_pos, s, t = generate_random_graph(X, Y, prob_type)

                    clusters_returned = None

                    if cluster_type == 'dense': 
                        clusters_returned = get_dense_clusters(G, X, Y, new_pos, len(H.nodes)-2,
                                                             start_node=s, end_node=t)
                    else:
                        clusters_returned = get_sparse_clusters(G, X, Y, new_pos, node_pos, len(H.nodes)-2,
                                                            start_node=s, end_node=t)

                    clusters[0] = [s]
                    for j, c in enumerate(clusters_returned.values()):
                        clusters[j + 1] = c.copy()
                    clusters[len(clusters)] = [t]

                    cluster_nodes = [list(c) for c in clusters.values()]
                    if cluster_type == 'dense': remove_edges_in_clusters(G, cluster_nodes)

                    budget_primary, budget_backup = compute_budget_simple(G, H, s, t, clusters, budget_delta) if cluster_type == 'dense' else compute_budget_hausdorff(G, H, clusters, budget_delta)
                    if budget_primary == float('inf'):
                        continue
                    capacities_primary = {e : 0 for e in G.edges}
                    capacities_backup = capacities_primary.copy()

                    # if i == 0:
                    #     draw_G(G,cluster_nodes,new_pos,X,Y,cluster_type,f'G{topology_number+1}_{cluster_type}.png')
                    # continue  

                    epsilon_primary = 5e-2
                    epsilon_backup = 1e-2

                    repairing_costs = {e : 0 for e in G.edges}
                    repairing_budget = {arc : 0 for arc in H.edges}

                    write_instance(f'{instance_prefix}_{i}',
                                   G, H, clusters, min(H.nodes), max(H.nodes),
                                   budget_primary, budget_backup,
                                   capacities_primary, capacities_backup,
                                   repairing_costs, repairing_budget, epsilon_primary, epsilon_backup)


if __name__ == "__main__":
    main()