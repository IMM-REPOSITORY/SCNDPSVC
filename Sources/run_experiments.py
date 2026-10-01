import argparse
import common
from common import ProblemData, UB, incumbent, models, remove_dangling_chains
from experiments import solver
from experiment_visualization import instance_output_dir, save_base_visualizations, save_experiment_solution
import networkx as nx
from parse_scndpsvc_instance import load_problem_data,  find_instance_files
from parallel_heuristic import init, getUB, getIncumbentSolution
from presolving import presolve_a_task

from optimization_models import prepare_budget_model, prepare_common_model, check_instance_feasibility, prepare_local_embedding_model, prepare_local_common_model

from time import sleep
import math

SC_PARAMS = ['budget', 'gluing', 'local_common', 'common']
SC_ORDERS = ['budget', 'gluing', 'local_common', 'common', 'all']

toy_pos = {
    0: (0, 0),
    1: (3, 2), 2: (3, -2),
    3: (5, 4),   4: (5, 3),
    5: (5, -3), 6: (5, -4),
    7: (8, 0), 8: (10, 0), 
    9: (1, 3),
    10: (4, 3),
    11: (7, 2),
    12: (8.5, 5),
    13: (8, -3)
}


def reconstruct_routes(data, x_solution, verbose=True):
    """
    Восстанавливает маршруты для каждой дуги (i,j) графа H по решению x_solution.
    x_solution – словарь {edge: 0/1}.
    Возвращает словарь: {(i,j): [список путей (каждый путь – список рёбер)]}.
    """
    G_full = data.G
    clusters = data.clusters
    H = data.H

    # Подграф, состоящий только из выбранных дуг
    active_edges = [e for e, val in x_solution.items() if val > 0.5]
    G_active = G_full.edge_subgraph(active_edges).copy()

    routes = {}
    for (i, j) in H.edges:
        routes[(i,j)] = []
        found = False
        for u in clusters[i]:
            if u not in G_active:
                continue
            for v in clusters[j]:
                if v not in G_active:
                    continue
                try:
                    # Ищем любой путь (кратчайший по стоимости) в выбранном подграфе
                    paths = nx.all_simple_paths(G_active, u, v, cutoff=max(data.budget_backup.values()))
                    if paths:
                        routes[(i,j)].extend(paths)
                        found = True
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        if verbose:
            if not found:
                print(f"Дуга H ({i},{j}) – нет маршрута в решении!")
            else:
                print(f"\nДуга H ({i},{j}):")
                print(f'{routes[(i,j)]=}')
    return routes


def print_problem_summary(data):
    print(f"Instance: {data.name_file.strip()}")
    print(f"Clusters: {data.clusters}")
    print(f"Hubs: {data.hubs}")
    print(f"Number of edges: {len(data.G.edges)}")
    print(f"Ordered pairs: {data.ordered_pairs}")
    print(
        f"start_cluster={data.start_cluster}, dest_cluster={data.dest_cluster}, "
        f"start_node={data.start_node}, dest_node={data.dest_node}"
    )
    print(f"budget_primary={data.budget_primary}, budget_backup={data.budget_backup}, epsilon_primary={data.epsilon_primary}, epsilon_backup={data.epsilon_backup}")
    print(f"separation order={data.separation_order}")
    print(f"constraints={data.run}")
    print(f"TIME_LIMIT={common.TIME_LIMIT}")

def compute_probs(G):
    summ = sum([(2 - attr['p']) / (attr['c'] + 1) for u,v,attr in G.edges(data=True)])
    probs = {(u,v): ((2 - attr['p']) / (attr['c'] + 1)) / summ for u, v, attr in G.edges(data=True)}
    return probs

def run_one(data, constraint_types, output_dir, dct, sync_event, debug=True, visualize=False):
    print(f"\n=== Experiment: {constraint_types} ===")
    result = solver(data, dct, sync_event, debug=debug)
    result.incumbent = dct['incumbent']
    print(f'{result.x_solution=}')
    print(f"Incumbent={dct['incumbent']}")
    if visualize:
        save_experiment_solution(data, result, constraint_types, output_dir, pos=toy_pos)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="Path to one .scndpsvc file or a directory with .scndpsvc files.",
    )
    parser.add_argument("-q", action="store_true", help="Disable per-iteration debug logs.")
    parser.add_argument("-v", action="store_true", help="Save PNG visualizations.")
    parser.add_argument(
        "-e",
        "--run",
        nargs='+',
        choices=['all'] + SC_PARAMS,
        default="all",
        help="Available SC parameters: budget, gluing, local_common, common."
    )
    parser.add_argument("--output-dir", default="figures", help="Directory for generated PNG visualizations.")
    parser.add_argument("-o", "--order", choices=SC_ORDERS, default='budget', help="Available ORDER parameters: budget, gluing, common")
    parser.add_argument("-l", "--logs", default="logs", help="A folder with log-files of the previous iteration")
    parser.add_argument("-t", "--time_limit", default="18000", help="Time Limit")
    parser.add_argument("-b", "--backup_delta", default="1.1", help="Backup delta")
    parser.add_argument("-p", "--prob_epsilon", default="0.01", help="epsilon threshold")
    return parser.parse_args()

def init_solver(data, constraint_types, output_dir, debug=True, visualize=False):
    global probs
    global models
    probs = compute_probs(data.G)
    solution = {e : 0 for e in data.G.edges}

    for type in constraint_types:
        if type == 'gluing':
            models['gluing_backup'] = {cluster_number: prepare_local_embedding_model(data, solution, cluster_number, with_probs=True) for cluster_number in data.H.nodes if not cluster_number in (data.start_cluster, data.dest_cluster)}

        if type == 'budget':
            models['budget_primary'] = {arc : prepare_budget_model(data, arc, solution) for arc in data.H.edges}
            models['budget_backup'] = {arc : prepare_budget_model(data, arc, solution, is_backup = True) for arc in data.H.edges}

        if type == 'local_common':
            models['local_common'] = {cluster_number: prepare_local_common_model(data, solution, cluster_number, is_backup=False) for cluster_number in data.H.nodes if not cluster_number in (data.start_cluster, data.dest_cluster)}
            models['local_common_backup'] = {cluster_number: prepare_local_common_model(data, solution, cluster_number, is_backup=True) for cluster_number in data.H.nodes if not cluster_number in (data.start_cluster, data.dest_cluster)}

        if type == 'common':
            models['common'] = prepare_common_model(data, solution, is_backup=False)
            models['common_backup'] = prepare_common_model(data, solution, is_backup=True)


def main():
    """Загрузить instance, запустить эксперименты и сохранить визуализации."""
    global UB
    global incumbent
    args = parse_args()
    constraint_types = list(SC_PARAMS) if 'all' in args.run else args.run
    debug = not args.q
    separation_order = args.order
    log_filename = args.logs
    backup_delta = float(args.backup_delta)
    epsilon = float(args.prob_epsilon)
    common.TIME_LIMIT = int(args.time_limit)

    instance_files = find_instance_files(args.path)
    if not instance_files:
        raise FileNotFoundError(f"No .scndpsvc files found under: {args.path}")

    for file_idx, instance_file in enumerate(instance_files, start=1):
        print(f"\n##### Instance {file_idx}/{len(instance_files)}: {instance_file} #####")
        data = load_problem_data(instance_file, log_filename, constraint_types)
        if data.budget_primary and data.budget_backup:
            data.budget_backup = {e : math.ceil(backup_delta * data.budget_primary[e]) for e in data.budget_primary}
        data.separation_order = separation_order
        data.run = args.run
        data.epsilon_backup = epsilon
        print_problem_summary(data)
        output_dir = instance_output_dir(args.output_dir, instance_file)
        if args.v:
            data.pos = toy_pos
            save_base_visualizations(data, output_dir)

        G, start_solution, UB = presolve_a_task(data, constraint_types)
        init_solver(data, constraint_types, output_dir, debug=debug, visualize=args.v)
        if check_instance_feasibility(data, start_solution, ct=constraint_types):
            print(f'Start solution found!')
            print(f'Upper bound {UB=}')
        else:
            print(f'Instance is not feasible!')
            exit(0)


        if data.mip_start and check_instance_feasibility(data, data.mip_start, ct=constraint_types):
            incumbent_cost = sum(data.G.edges[e]['c'] for e, v in data.mip_start.items() if v > 0.5)
            print(f'The MIP start provided an incumbent of cost {incumbent_cost}')
            UB = incumbent_cost
            incumbent = data.mip_start.copy()
            improved_incumbent = remove_dangling_chains(data, incumbent)
            incumbent = improved_incumbent.copy()
            improved_incumbent_cost = sum(data.G.edges[e]['c'] for e, v in improved_incumbent.items() if v > 0.5)
            UB = improved_incumbent_cost
        elif data.mip_start:
            print(f'The provided MIP start is not valid!')



        ####  start parallel heuristic
        gs_pool, rins_pool, dct, new_incumbent_event = init(data, start_solution, UB)
        sleep(3)


        # for experiment in experiments:
        run_one(data, constraint_types, output_dir, dct, new_incumbent_event, debug=debug, visualize=args.v)


if __name__ == "__main__":
    main()
