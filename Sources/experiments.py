import common
from common import UB, incumbent, models, is_time_limit_exceeded

from dataclasses import dataclass
from time import perf_counter

import gurobipy as gp
from gurobipy import GRB

from optimization_models import  prepare_main_model, updateXconstraint, check_instance_feasibility
from parallel_heuristic import getUB, getIncumbentSolution

import multiprocessing as mp
import time
import os

@dataclass
class ExperimentResult:
    master_model: object
    x_solution: dict
    iterations: int
    cuts_added: int
    cuts_without_probs: int
    cuts_with_probs: int
    is_time_limit_exceeded: bool
    appended_master_constrs: list

FEASIBLE_POOL = []
NUM_FEASIBLE = 100

CURRENT_DATA = None


def _x_solution(master_model):
    return {e: var.X for e, var in master_model._x.items()}


def _print_init(master_model, debug):
    pass

def _solution_cost(data, sol):
    return sum(data.G.edges[e]['c'] for e, v in sol.items() if v == 1)

def on_new_incumbent(model, where):
    global CURRENT_DATA, FEASIBLE_POOL
    if where == GRB.Callback.MIPSOL:
        if CURRENT_DATA is None:
            return
        
        vars_w = model.cbGetSolution(model._vars_w)
        x_vals = {}
        for e in CURRENT_DATA.G.edges:
            if e in vars_w:
                x_vals[e] = vars_w[e]
            else:
                x_vals[e] = 0
        
        if len(FEASIBLE_POOL) < NUM_FEASIBLE:
            FEASIBLE_POOL.append(x_vals)


def _optimize_with_timer(model, m_type, elapsed_time):
    start_time = perf_counter()
    remaining = max(1, int(common.TIME_LIMIT - elapsed_time))
    model.setParam("TimeLimit", remaining)
    model.optimize(on_new_incumbent)    
    return perf_counter() - start_time


def _print_table_header(debug):
    if debug:
        print(
            f"{'elapsed_s':>10}  {'iter':>5}  {'BP':>4}  {'BB':>4}  "
            f"{'GL':>4} {'CP':>4} {'CB':>4} {'RP':>4} {'LC':>4} {'LCB':>4} {'COM':>4} {'COM_B':>4} {'total_cuts':>10}  {'master_obj':>12} {'UB':>14} {'gap (%)':>16}"
        )
        print(
            f"{'-' * 10}  {'-' * 5}  {'-' * 4}  {'-' * 4} {'-' * 4} {'-' * 4} {'-' * 4} "
            f"{'-' * 4} {'-' * 4} {'-' * 4} {'-' * 10} {'-' * 10} {'-' * 10}  {'-' * 12} {'-' * 14} {'-' * 16}"
        )


def _print_iteration_summary(
    it,
    solve_start_time,
    cuts_statistics,
    cuts_added,
    master_model,
    debug,
):
    global UB
    UB = getUB()
    LB = master_model.objVal

    if debug:
        elapsed = perf_counter() - solve_start_time
        print(
            f"{elapsed:10.3f}  {it:5d}  {cuts_statistics.get('budget_primary',0):4d}  "
            f"{cuts_statistics.get('budget_backup',0):4d} {cuts_statistics.get('gluing_backup',0):4d} "
            f"{cuts_statistics.get('capacity_primary',0):4d} {cuts_statistics.get('capacity_backup',0):4d} {cuts_statistics.get('repairing',0):4d} {cuts_statistics.get('local_common',0):4d} {cuts_statistics.get('local_common_backup',0):4d} {cuts_statistics.get('common',0):4d} {cuts_statistics.get('common_backup',0):4d}"
            f"{cuts_added:10d}  {LB:12.6g}  {UB:12.6g} {(UB-LB)/LB:12.1%}"
        )


def _print_final(it, solve_start_time, cuts_statistics, total_cuts_added, master_model, all_branching_nodes_explored=False):
    # global UB
    global is_time_limit_exceeded
    print("Summary")
    if all_branching_nodes_explored: print(f'All cutting planes are explored!')
    print(f"Iterations: {it}")
    print(f"budget_primary cuts: {cuts_statistics.get('budget_primary', 0)}")
    print(f"budget_backup cuts: {cuts_statistics.get('budget_backup', 0)}")
    print(f"gluing cuts: {cuts_statistics.get('gluing_backup', 0)}")
    print(f"capacity_primary cuts: {cuts_statistics.get('capacity_primary', 0)}")
    print(f"capacity_backup cuts: {cuts_statistics.get('capacity_backup', 0)}")
    print(f"repairing cuts: {cuts_statistics.get('repairing', 0)}")
    print(f"local_common cuts: {cuts_statistics.get('local_common', 0)}")
    print(f"local_common_backup cuts: {cuts_statistics.get('local_common_backup', 0)}")
    print(f"common cuts: {cuts_statistics.get('common', 0)}")
    print(f"common_backup cuts: {cuts_statistics.get('common_backup', 0)}")
    print(f"Total cuts: {total_cuts_added}")
    print(f"Master obj.: {master_model.objVal}")
    if is_time_limit_exceeded: print(f'Time limit exceeded!')
    print(f"Total time (s): {perf_counter() - solve_start_time:.3f}")
    print(f"UB={UB}")
    if is_time_limit_exceeded: print(f"GAP={(UB-master_model.objVal)/master_model.objVal:.1%}")
    if not is_time_limit_exceeded: print(f"GAP=0.0%")


def _make_result(master_model, it, cuts_without_probs, cuts_with_probs, cuts_added, is_time_limit_exceeded, appended_master_constrs):
    return ExperimentResult(
        master_model=master_model,
        x_solution=_x_solution(master_model),
        iterations=it,
        cuts_added=cuts_added,
        cuts_without_probs=cuts_without_probs,
        cuts_with_probs=cuts_with_probs,
        is_time_limit_exceeded=is_time_limit_exceeded,
        appended_master_constrs=appended_master_constrs
    )


def check_feasibility_problem(w_edges, msg):
    if len(w_edges) == 0:
        print(f'Instance is infeasible: ' + msg)
        exit(0)


def generate_model_order(data):
    models_order = []
    if data.separation_order == 'budget':
        if 'budget_primary' in models and 'budget_backup' in models: models_order.append([(f'budget_primary', models[f'budget_primary']), (f'budget_backup', models[f'budget_backup'])])
        if 'gluing_backup' in models: models_order.append([(f'gluing_backup', models[f'gluing_backup'])])
        if 'local_common' in models: models_order.append([(f'local_common', models[f'local_common'])])
        if 'local_common_backup' in models: models_order.append([(f'local_common_backup', models[f'local_common_backup'])])
        if 'common' in models: models_order.append([(f'common', models[f'common'])])
        if 'common_backup' in models: models_order.append([(f'common_backup', models[f'common_backup'])])
    elif data.separation_order == 'gluing':
        if 'gluing_backup' in models: models_order.append([(f'gluing_backup', models[f'gluing_backup'])])
        if 'budget_primary' in models and 'budget_backup' in models: models_order.append([(f'budget_primary', models[f'budget_primary']), (f'budget_backup', models[f'budget_backup'])])
        if 'local_common' in models: models_order.append([(f'local_common', models[f'local_common'])])
        if 'local_common_backup' in models: models_order.append([(f'local_common_backup', models[f'local_common_backup'])])
        if 'common' in models: models_order.append([(f'common', models[f'common'])])
        if 'common_backup' in models: models_order.append([(f'common_backup', models[f'common_backup'])])
    elif data.separation_order == 'local_common':
        if 'local_common' in models: models_order.append([(f'local_common', models[f'local_common'])])
        if 'local_common_backup' in models: models_order.append([(f'local_common_backup', models[f'local_common_backup'])])
        if 'budget_primary' in models and 'budget_backup' in models: models_order.append([(f'budget_primary', models[f'budget_primary']), (f'budget_backup', models[f'budget_backup'])])
        if 'gluing_backup' in models: models_order.append([(f'gluing_backup', models[f'gluing_backup'])])
        if 'common' in models: models_order.append([(f'common', models[f'common'])])
        if 'common_backup' in models: models_order.append([(f'common_backup', models[f'common_backup'])])
    elif data.separation_order == 'common':
        if 'common' in models: models_order.append([(f'common', models[f'common'])])
        if 'common_backup' in models: models_order.append([(f'common_backup', models[f'common_backup'])])
    elif data.separation_order == 'all':
        if 'gluing_backup' in models and 'budget_backup' in models and 'common_backup' in models: models_order.append([(f'budget_primary', models[f'budget_primary']), (f'budget_backup', models[f'budget_backup']), (f'gluing_backup', models[f'gluing_backup'])])
        if 'local_common' in models: models_order.append([(f'local_common', models[f'local_common'])])
        if 'local_common_backup' in models: models_order.append([(f'local_common_backup', models[f'local_common_backup'])])
        if 'common' in models: models_order.append([(f'common', models[f'common'])])
        if 'common_backup' in models: models_order.append([(f'common_backup', models[f'common_backup'])])

    return models_order

def greedy_remove_edges(data, sol, sort_mode='cost'):
    edges = list(data.G.edges(data=True))
    if sort_mode == 'cost':
        edges.sort(key=lambda e: e[2]['c'], reverse=True)
    elif sort_mode == 'random':
        std_random.shuffle(edges)
    elif sort_mode == 'cost_over_reliability':
        edges.sort(key=lambda e: e[2]['c'] / (1.0 - e[2]['p']), reverse=True)
    else:
        raise ValueError(f"Unknown sort_mode: {sort_mode}")

    batch_size = max(1, len(edges) // 20)
    number_improved_blocks = 0
    i = 0
    while i < len(edges):
        batch = []
        for j in range(i, min(i + batch_size, len(edges))):
            u, v, _ = edges[j]
            if sol[(u, v)] == 1:
                batch.append((u, v))
        test = sol.copy()
        for u, v in batch:
            test[(u, v)] = 0
        if _cached_feasibility(data, test):
            sol = test
            i += batch_size
        else:
            for u, v in batch:
                test2 = sol.copy()
                test2[(u, v)] = 0
                if _cached_feasibility(data, test2):
                    sol = test2
                    number_improved_blocks += 1
                    break
            i += batch_size
    return sol


def solver(data, dct, sync_event=None, debug=True):
    global UB
    global incumbent
    global is_time_limit_exceeded
    global CURRENT_DATA, FEASIBLE_POOL
    CURRENT_DATA = data
    appended_master_constrs = []
    master_model = prepare_main_model(data.G, data.appended_master_constrs)
    master_model.setParam("Threads", 3)
    master_model.optimize()

    it = 1
    total_cuts_added = 0
    _print_init(master_model, debug)
    solve_start_time = perf_counter()
    _print_table_header(debug)

    start_time = time.time()
    # is_time_limit_exceeded = False

    cuts_statistics = {m_type : 0 for m_type in models.keys()}

    models_order = generate_model_order(data)
    all_branching_nodes_explored = False

    while True:
        if getUB() - master_model.objVal <= 1e-6:
            break
        if time.time() - start_time >= common.TIME_LIMIT:
            is_time_limit_exceeded = True
            break
        x_sol_dict = _x_solution(master_model)
        dct['obj_vector'] = x_sol_dict.copy()

        need_solve_master = False
        iteration_sep_time = 0.0

        #iterating through all initialized models
        for model_lst in models_order:
            for m_tuple in model_lst:
                m_type = m_tuple[0]
                m_obj = m_tuple[1]
                if m_type in ('gluing_backup', 'local_common', 'local_common_backup'):
                    iteration_edges = list(set(data.H.nodes) - set([data.start_cluster]) - set([data.dest_cluster]))
                elif m_type in ('budget_primary', 'budget_backup'):
                    iteration_edges = data.H.edges
                else:
                    iteration_edges = [m_type]

                # iteration_edges = data.H.edges() if isinstance(m_obj, dict) else [m_type]
                inner_cuts_number = 0
                for ie in iteration_edges:
                    model = m_obj if ie in ('common', 'common_backup') else m_obj[ie]
                    updateXconstraint(model, data.G, x_sol_dict)

                    sep_time = _optimize_with_timer(model, m_type, time.time() - start_time)

                    if time.time() - start_time >= common.TIME_LIMIT:
                        is_time_limit_exceeded = True
                        break

                    iteration_sep_time += sep_time
                    if model.status == GRB.INFEASIBLE:
                        continue

                    if (model.status == GRB.OPTIMAL):
                        need_solve_master = True
                        optimal_cut = [e for e, var in model._vars_w.items()]
                        w_edges = [e for e, var in model._vars_w.items() if var.X > 0.5]

                        check_feasibility_problem(w_edges, f'there is no solution:: {m_type} restrictions are violated')

                        master_model.addConstr(gp.quicksum([master_model._x[e] for e in w_edges]) >= 1, name=f'{m_type}')
                        appended_master_constrs.append([e for e in w_edges])

                        for x_sol in FEASIBLE_POOL:
                            if x_sol == optimal_cut:
                                continue
                            master_model.addConstr(gp.quicksum(master_model._x[e] for e in x_sol if x_sol[e] > 0.5) >= 1)
                            appended_master_constrs.append([e for e in x_sol if x_sol[e] > 0.5])


                        added_cuts = len(FEASIBLE_POOL) if optimal_cut in FEASIBLE_POOL else len(FEASIBLE_POOL) + 1
                        total_cuts_added += added_cuts
                        inner_cuts_number += added_cuts

                        # total_cuts_added += 1 + len(FEASIBLE_POOL)
                        # inner_cuts_number += 1 + len(FEASIBLE_POOL)

                        FEASIBLE_POOL = []

                cuts_statistics[m_type] += inner_cuts_number

            if need_solve_master:
                master_model.optimize()
                _print_iteration_summary(
                    it,
                    solve_start_time,
                    cuts_statistics,
                    total_cuts_added,
                    master_model,
                    debug,
                )
                it += 1
                break

        if not need_solve_master and not is_time_limit_exceeded:
            all_branching_nodes_explored = True
            break

    _print_final(it, solve_start_time, cuts_statistics, total_cuts_added, master_model, all_branching_nodes_explored)
    return _make_result(master_model, it, 0, 0, total_cuts_added, is_time_limit_exceeded, appended_master_constrs)