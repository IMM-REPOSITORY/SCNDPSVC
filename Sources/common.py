from dataclasses import dataclass
import networkx as nx
import copy


UB = float('inf')
incumbent = {}

TIME_LIMIT = 36000

models = {}
probs = {}

BUNCH_SIZE = 1

is_time_limit_exceeded = False

def remove_dangling_chains(data, sol):
    G_full = data.G
    start_node = data.start_node
    dest_node = data.dest_node

    cleaned_sol = sol.copy()
    changed = True

    while changed:
        changed = False
        active_edges = [e for e, val in cleaned_sol.items() if val == 1]
        if not active_edges:
            break

        G_active = G_full.edge_subgraph(active_edges).copy()

        dangling = []
        for node in G_active.nodes:
            if node == start_node or node == dest_node:
                continue
            if G_active.degree(node) == 1:
                dangling.append(node)

        if not dangling:
            break

        for node in dangling:
            edges_to_remove = list(G_active.edges(node))
            for e in edges_to_remove:
                if cleaned_sol.get(e, 0) == 1:
                    cleaned_sol[e] = 0
                    changed = True
    return cleaned_sol

@dataclass
class ProblemData:
    name_file: str
    nodes_number: int
    graph_weights: dict
    graph_probs: dict
    graph_capacities_primary: dict
    graph_capacities_backup: dict
    graph_budgets: dict
    graph_repairing_costs: dict
    clusters: dict
    clusters_order: dict
    hubs: list
    start_cluster: int
    dest_cluster: int
    budget_primary: dict
    budget_backup: dict
    repairing_budget: dict
    epsilon_primary: float
    epsilon_backup: float
    G: nx.DiGraph
    H: nx.DiGraph
    ordered_pairs: list
    start_node: int
    dest_node: int
    pos: dict
    h_pos: dict
    mip_start: dict
    appended_master_constrs: list
    separation_order: str
    run: dict