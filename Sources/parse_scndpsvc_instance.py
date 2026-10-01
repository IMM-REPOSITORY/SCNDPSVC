import networkx as nx
import numpy as np
import ast
import re
from pathlib import Path
import os
from common import ProblemData, TIME_LIMIT

def parse_solution_from_routes_log(log_file_path, G):
    incumbent = {e : 1 for e in G.edges}
    appended_master_constrs = []
    all_branching_nodes_explored = False
    time_limit_exceeded = True
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'All branching nodes' in line or 'All cutting planes' in line:
                all_branching_nodes_explored = True
            if 'Total time (s)' in line:
                time_limit_exceeded = True if float(line.split(':')[1]) > TIME_LIMIT - 1 else False
            # if 'Incumbent' or 'result.x_solution' in line:
            if 'Incumbent' in line and (time_limit_exceeded or not all_branching_nodes_explored):
                incumbent = ast.literal_eval(line.split('=')[1])
            elif 'result.x_solution' in line and not time_limit_exceeded and all_branching_nodes_explored:
                incumbent = ast.literal_eval(line.split('=')[1])
            if 'result.appended_master_constrs' in line:
                appended_master_constrs = ast.literal_eval(line.split('=')[1])

    return incumbent, appended_master_constrs

def remove_negative_values(d: dict):
    return {key : val for key, val in d.items() if val >= 0 and key[0] != key[1]}

def parse_row_data(row, func):
    return func(row.split(':')[-1].strip('\n'))

def parse_mip_start(mip_filename):
    mip_start = {}
    with open(mip_filename, 'r') as mip_file:
        row = mip_file.readline()
        mip_start = ast.literal_eval(row.split('=')[1])
    return mip_start

def parse_instance_scndpsvc(file_name):
    graph_capacities_primary = None
    graph_capacities_backup = None
    graph_budgets = None
    graph_repairing_costs = None
    budget_primary = None
    budget_backup = None
    repairing_budget = None
    epsilon_primary = None
    epsilon_backup = None
    L1 = None
    L2 = None
    epsilon = None   # старый единый epsilon

    with open(file_name, 'r') as file:
        row = file.readline()
        while row != '':
            if 'NAME' in row:
                name_file = parse_row_data(row, str)
            if 'DIMENSION' in row:
                nodes_number = parse_row_data(row, int)
            if 'EDGE_WEIGHT_SECTION' in row:
                row = file.readline()
                graph_weights = remove_negative_values(ast.literal_eval(row))
            if 'EDGE_PROB_SECTION' in row:
                row = file.readline()
                graph_probs = remove_negative_values(ast.literal_eval(row))
            if 'EDGE_CAPACITY_SECTION_PRIMARY' in row:
                row = file.readline()
                graph_capacities_primary = remove_negative_values(ast.literal_eval(row))
            if 'EDGE_CAPACITY_SECTION_BACKUP' in row:
                row = file.readline()
                graph_capacities_backup = remove_negative_values(ast.literal_eval(row))
            if 'EDGE_BUDGET_SECTION' in row:
                row = file.readline()
                graph_budgets = remove_negative_values(ast.literal_eval(row))
            if 'EDGE_REPAIRING_SECTION' in row:
                row = file.readline()
                graph_repairing_costs = remove_negative_values(ast.literal_eval(row))
            if 'CLUSTERS_SETS_SECTION' in row:
                row = file.readline()
                clusters_set = ast.literal_eval(row)
            if 'CLUSTERS_SETS_ORDERING' in row:
                row = file.readline()
                clusters_order = ast.literal_eval(row)
            if 'HUBS_SECTION' in row:
                row = file.readline()
                hubs = ast.literal_eval(row)
            if 'START_GROUP_SECTION' in row:
                row = file.readline()
                start_cluster = parse_row_data(row, int)
            if 'DEST_GROUP_SECTION' in row:
                row = file.readline()
                dest_cluster = parse_row_data(row, int)
            if 'LENGTH1_SECTION' in row:
                row = file.readline()
                L1 = parse_row_data(row, int)
            if 'LENGTH2_SECTION' in row:
                row = file.readline()
                L2 = parse_row_data(row, int)
            if 'BUDGET_SECTION_PRIMARY' in row:
                row = file.readline()
                budget_primary = ast.literal_eval(row)
            if 'BUDGET_SECTION_BACKUP' in row:
                row = file.readline()
                budget_backup = ast.literal_eval(row)
            if 'REPAIRING_BUDGET' in row:
                row = file.readline()
                repairing_budget = ast.literal_eval(row)
            if 'EPSILON_SECTION_PRIMARY' in row:
                row = file.readline()
                epsilon_primary = parse_row_data(row, float)
            if 'EPSILON_SECTION_BACKUP' in row:
                row = file.readline()
                epsilon_backup = parse_row_data(row, float)
            if 'EPSILON_SECTION' in row and 'EPSILON_SECTION_PRIMARY' not in row:  # старый формат
                row = file.readline()
                epsilon = parse_row_data(row, float)
            row = file.readline()

    if L1 is not None:
        if graph_budgets is None:
            graph_budgets = {e: 1 for e in graph_weights}
        H_temp_edges = []
        for from_cl, to_list in clusters_order.items():
            for to_cl in to_list:
                H_temp_edges.append((from_cl, to_cl))
        if budget_primary is None:
            budget_primary = {arc: L1 for arc in H_temp_edges}
        if budget_backup is None:
            budget_backup = {arc: L2 for arc in H_temp_edges}
        if epsilon_primary is None and epsilon is not None:
            epsilon_primary = epsilon
            epsilon_backup = epsilon
        elif epsilon_primary is None:
            epsilon_primary = 1.0   # не активно
            epsilon_backup = 1.0
        if graph_capacities_primary is None:
            graph_capacities_primary = {}
        if graph_capacities_backup is None:
            graph_capacities_backup = {}
        if graph_repairing_costs is None:
            graph_repairing_costs = {}
        if repairing_budget is None:
            repairing_budget = {arc: 0 for arc in H_temp_edges}

    return (name_file, nodes_number, graph_weights, graph_probs,
            graph_capacities_primary, graph_capacities_backup,
            graph_budgets, graph_repairing_costs,
            clusters_set, clusters_order, hubs,
            start_cluster, dest_cluster,
            budget_primary, budget_backup,
            repairing_budget, epsilon_primary, epsilon_backup)

def parse_instance_scndpsvc_actual(file_name):
    with open(file_name, 'r') as file:
        row = file.readline()
        while row != '':
            if 'NAME' in row:
                name_file = parse_row_data(row, str)
            if 'DIMENSION' in row:
                nodes_number = parse_row_data(row, int)
            if 'EDGE_WEIGHT_SECTION' in row:
                # graph_weights = parse_matrix(nodes_number, file)
                row = file.readline()
                # graph_weights = ast.literal_eval(row)
                graph_weights = remove_negative_values(ast.literal_eval(row))
            if 'EDGE_PROB_SECTION' in row:
                # graph_probs = parse_matrix(nodes_number, file)
                row = file.readline()
                # graph_probs = ast.literal_eval(row)
                graph_probs = remove_negative_values(ast.literal_eval(row))
            if 'EDGE_CAPACITY_SECTION_PRIMARY' in row:
                row = file.readline()
                graph_capacities_primary = remove_negative_values(ast.literal_eval(row))
            if 'EDGE_CAPACITY_SECTION_BACKUP' in row:
                row = file.readline()
                graph_capacities_backup = remove_negative_values(ast.literal_eval(row))
            if 'EDGE_BUDGET_SECTION' in row:
                row = file.readline()
                graph_budgets = remove_negative_values(ast.literal_eval(row))
            if 'EDGE_REPAIRING_SECTION' in row:
                row = file.readline()
                graph_repairing_costs = remove_negative_values(ast.literal_eval(row))
            if 'CLUSTERS_SETS_SECTION' in row:
                row = file.readline()
                clusters_set = ast.literal_eval(row)
            if 'CLUSTERS_SETS_ORDERING' in row:
                row = file.readline()
                clusters_order = ast.literal_eval(row)
            if 'HUBS_SECTION' in row:
                row = file.readline()
                hubs = ast.literal_eval(row)
            if 'START_GROUP_SECTION' in row:
                row = file.readline()
                start_cluster = parse_row_data(row, int)
            if 'DEST_GROUP_SECTION' in row:
                row = file.readline()
                dest_cluster = parse_row_data(row, int)
            if 'BUDGET_SECTION_PRIMARY' in row:
                row = file.readline()
                budget_primary = ast.literal_eval(row)
            if 'BUDGET_SECTION_BACKUP' in row:
                row = file.readline()
                budget_backup = ast.literal_eval(row)
            if 'REPAIRING_BUDGET' in row:
                row = file.readline()
                # repairing_budget = parse_row_data(row, int)
                repairing_budget = ast.literal_eval(row)
            if 'EPSILON_SECTION_PRIMARY' in row:
                row = file.readline()
                epsilon_primary = parse_row_data(row, float)
            if 'EPSILON_SECTION_BACKUP' in row:
                row = file.readline()
                epsilon_backup = parse_row_data(row, float)
            row = file.readline()
    return name_file, nodes_number, graph_weights, graph_probs, graph_capacities_primary, graph_capacities_backup, graph_budgets, graph_repairing_costs, clusters_set, clusters_order, hubs, start_cluster, dest_cluster, budget_primary, budget_backup, repairing_budget, epsilon_primary, epsilon_backup
            

def topological_generations(G):
    if not G.is_directed():
        raise nx.NetworkXError("Topological generations not defined for undirected graphs.")
    indegree_map = dict(G.in_degree())
    zero_indegree = [v for v, d in indegree_map.items() if d == 0]
    while zero_indegree:
        yield zero_indegree
        new_zero_indegree = []
        for v in zero_indegree:
            for child in G.successors(v):
                indegree_map[child] -= 1
                if indegree_map[child] == 0:
                    new_zero_indegree.append(child)
        zero_indegree = new_zero_indegree


def build_layouts(G, H, clusters, hubs):
    cluster_layers = {}
    for layer_idx, cluster_ids in enumerate(topological_generations(H)):
        for cluster_id in cluster_ids:
            cluster_layers[cluster_id] = layer_idx

    cluster_by_node = {}
    for cluster_id, vertices in clusters.items():
        for node in vertices:
            cluster_by_node[node] = cluster_id

    x_gap = 4.2
    cluster_gap = 2.8
    vertex_gap = 0.75

    pos = {}
    clusters_by_layer = {}
    for cluster_id, layer in cluster_layers.items():
        clusters_by_layer.setdefault(layer, []).append(cluster_id)

    cluster_centers = {}
    for layer, cluster_ids in clusters_by_layer.items():
        ordered_clusters = sorted(cluster_ids)
        for cluster_idx, cluster_id in enumerate(ordered_clusters):
            center_y = ((len(ordered_clusters) - 1) / 2 - cluster_idx) * cluster_gap
            cluster_centers[cluster_id] = (layer * x_gap, center_y)
            vertices = sorted(clusters[cluster_id])
            for vertex_idx, node in enumerate(vertices):
                y = center_y + ((len(vertices) - 1) / 2 - vertex_idx) * vertex_gap
                pos[node] = (layer * x_gap, y)

    max_layer = max(cluster_layers.values(), default=0)
    for hub in sorted(hubs):
        predecessor_layers = []
        successor_layers = []
        neighbor_ys = []

        for pred in G.predecessors(hub):
            if pred in cluster_by_node:
                cluster_id = cluster_by_node[pred]
                predecessor_layers.append(cluster_layers[cluster_id])
                neighbor_ys.append(pos[pred][1])
        for succ in G.successors(hub):
            if succ in cluster_by_node:
                cluster_id = cluster_by_node[succ]
                successor_layers.append(cluster_layers[cluster_id])
                neighbor_ys.append(pos[succ][1])

        if predecessor_layers and successor_layers:
            hub_layer = (max(predecessor_layers) + min(successor_layers)) / 2
        elif predecessor_layers:
            hub_layer = min(max(predecessor_layers) + 0.5, max_layer)
        elif successor_layers:
            hub_layer = max(min(successor_layers) - 0.5, 0)
        else:
            hub_layer = max_layer / 2

        x = hub_layer * x_gap
        y = sum(neighbor_ys) / len(neighbor_ys) if neighbor_ys else 0.0
        pos[hub] = (x, y)

    for node in G.nodes:
        pos.setdefault(node, (max_layer * x_gap / 2, 0.0))

    h_pos = {cluster_id: cluster_centers[cluster_id] for cluster_id in H.nodes if cluster_id in cluster_centers}
    return pos, h_pos

def remove_isolated_vertices(G, clusters, hubs, start_node, dest_node):
    isolated = []
    for i in clusters:
        for v in clusters[i]:
            if v not in G.nodes:
                isolated.append(v)
    

    # Обновляем кластеры
    for c_id in list(clusters.keys()):
        clusters[c_id] = [v for v in clusters[c_id] if v not in isolated]
        if not clusters[c_id]:  # если кластер опустел, удаляем его
            del clusters[c_id]

    # Обновляем хабы
    hubs = [v for v in hubs if v not in isolated]

    return G, clusters, hubs

def load_problem_data(instance_file, log_filename, constraint_types, need_to_build_layouts = True):
    (
        name_file,
        nodes_number,
        graph_weights,
        graph_probs,
        graph_capacities_primary,
        graph_capacities_backup,
        graph_budgets,
        graph_repairing_costs,
        clusters,
        clusters_order,
        hubs,
        start_cluster,
        dest_cluster,
        budget_primary,
        budget_backup,
        repairing_budget,
        epsilon_primary,
        epsilon_backup
    ) = parse_instance_scndpsvc(instance_file)

    G = nx.DiGraph()
    for (u, v), cost in graph_weights.items():
        G.add_edge(u, v, c=cost, p=graph_probs[(u, v)], b=graph_budgets[(u,v)])

    H = nx.DiGraph()
    for cluster_from, clusters_to in clusters_order.items():
        for cluster_to in clusters_to:
            H.add_edge(cluster_from, cluster_to)

    ordered_pairs = list(H.edges())
    start_node = clusters[start_cluster][0]
    dest_node = clusters[dest_cluster][0]

    if need_to_build_layouts:
        pos, h_pos = build_layouts(G, H, clusters, hubs)
    else:
        pos, h_pos = 0, 0

    if not 'capacity' in constraint_types:
        graph_capacities_primary = {e : float('inf') for e in graph_capacities_primary}
        graph_capacities_backup = {e : float('inf') for e in graph_capacities_backup}

    if not 'budget' in constraint_types and not 'common' in constraint_types and not 'local_common' in constraint_types:
        budget_primary = {e: len(G.nodes) for e in budget_primary}
        budget_backup = {e: len(G.nodes) for e in budget_backup}

    master_solution = {}
    mip_start = {}
    appended_master_constrs = []

    if Path(log_filename).is_file():
        # print(f'HERE!')
        mip_start, appended_master_constrs = parse_solution_from_routes_log(log_filename, G)

    G, clusters, hubs = remove_isolated_vertices(G, clusters, hubs, start_node, dest_node)

    return ProblemData(
        name_file=name_file,
        nodes_number=nodes_number,
        graph_weights=graph_weights,
        graph_probs=graph_probs,
        graph_capacities_primary=graph_capacities_primary,
        graph_capacities_backup=graph_capacities_backup,
        graph_budgets=graph_budgets,
        graph_repairing_costs=graph_repairing_costs,
        clusters=clusters,
        clusters_order=clusters_order,
        hubs=hubs,
        start_cluster=start_cluster,
        dest_cluster=dest_cluster,
        budget_primary=budget_primary,
        budget_backup=budget_backup,
        repairing_budget=repairing_budget,
        epsilon_primary=epsilon_primary,
        epsilon_backup=epsilon_backup,
        G=G,
        H=H,
        ordered_pairs=ordered_pairs,
        start_node=start_node,
        dest_node=dest_node,
        pos=pos,
        h_pos=h_pos,
        mip_start=mip_start,
        appended_master_constrs=appended_master_constrs,
        separation_order='common',
        run='all'
    )
            
def find_instance_files(path):
    path = Path(path)
    if path.is_file():
        if path.suffix != ".scndpsvc":
            raise ValueError(f"Expected a .scndpsvc file, got: {path}.")
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.scndpsvc"))
    raise FileNotFoundError(f"Instance path does not exist: {path}")

