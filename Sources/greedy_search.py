import numpy as np
import random as std_random
import math
import copy
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import os, time
import random
import multiprocessing as mp

from common import *
from optimization_models import check_instance_feasibility, prepare_multicommodity_model

def updateUB(G, current_solution, callback, verbose, sync_event, msg):
    global UB, incumbent
    newUB = sum(c for u, v, c in G.edges(data='c')
                if (u, v) in current_solution and current_solution[(u, v)] > 0.5)
    if callback:
        callback(newUB, current_solution.copy(), verbose, sync_event)
    elif UB > newUB:
        UB = newUB
        incumbent = current_solution.copy()
        if verbose:
            print(f'UB changed by {msg}: {UB}')

def _z_solution(master_model):
    return {e: var.X for e, var in master_model._z.items() if var.X > 0.5}

def _solution_cost(data, sol):
    return sum(data.G.edges[e]['c'] for e, v in sol.items() if v > 0.5)

def append_embedding(constructed_solution, z_solution):
    for e in z_solution:
        if z_solution[e] > 0.5:
            constructed_solution[e] = 1
    return constructed_solution

_feasibility_cache = {}

def _cached_feasibility(data, sol):
    key = frozenset(e for e, v in sol.items() if v > 0.5)
    if key not in _feasibility_cache:
        _feasibility_cache[key] = check_instance_feasibility(data, sol)

    return _feasibility_cache[key]


def remove_edges(data, sol, removed_percent=0.1):
    edges_in = [e for e in data.G.edges(data=True) if sol[e[0],e[1]] > 0.5]
    need_sorting = std_random.randint(0, 10)
    sort_mode = 'random' if need_sorting >= 7 else 'cost'
    num_to_remove = int(len(edges_in) * removed_percent)

    if sort_mode == 'cost':
        edges_in.sort(key=lambda e: e[2]['c'], reverse=True)
        removed = edges_in[:num_to_remove]
    elif sort_mode == 'random':
        removed = random.sample(edges_in, num_to_remove)
    
    for e in removed:
        sol[e[0],e[1]] = 0

    return sol

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
            sol = test.copy()
            i += batch_size
        else:
            for u, v in batch:
                test2 = sol.copy()
                test2[(u, v)] = 0
                if _cached_feasibility(data, test2):
                    sol = test2.copy()
                    number_improved_blocks += 1
                    break
            i += batch_size
    return sol

def greedy_search(data, start_solution, updateCallback=None, verbose=True, dct=None, sync_event=None):
    std_random.seed(os.getpid() + int(time.time() * 1000))
    model_primary = prepare_multicommodity_model(data, is_backup = False)
    model_backup = prepare_multicommodity_model(data, is_backup = True)

    constructed_solution = start_solution
    constructed_solution = greedy_remove_edges(data, constructed_solution, sort_mode='random')

    MAX_STAGNATION = 30
    best_ub = float('inf')
    best_sol = None
    model_primary.optimize()
    if model_primary.status != GRB.OPTIMAL:
        print(f"Primary model is infeasible!")
        print(f"{model_primary.status=}")
        exit(0)
        
    z_solution = _z_solution(model_primary)
    sync_counter = 0
    bunch_size = int(len(data.G.edges) / 5)
    initial_bunch_size = bunch_size
    stagnation_counter = 0

    while True:
        moving_from_local_optima = False
        i = 0
        if dct and dct['UB'] < best_ub:
            best_ub = dct['UB']
            best_sol = dct['incumbent'].copy()
            constructed_solution = dct['incumbent'].copy()

        if check_instance_feasibility(data, constructed_solution):
            need_sorting = std_random.randint(0, 10)
            sort_mode = 'random' if need_sorting >= 9 else 'cost_over_reliability' if 7 <= need_sorting <= 8 else 'cost'
            constructed_solution = greedy_remove_edges(data, constructed_solution, sort_mode=sort_mode)
            constructed_solution = remove_dangling_chains(data, constructed_solution)
            constructed_solution_cost = _solution_cost(data, constructed_solution)
            if check_instance_feasibility(data, constructed_solution) and constructed_solution_cost < best_ub:
                updateUB(data.G, constructed_solution, updateCallback, True, sync_event, f'New incumbent was found at cost {constructed_solution_cost=}')
                best_ub = constructed_solution_cost
                best_sol = constructed_solution.copy()
                bunch_size = 1
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                bunch_size += 1

        if stagnation_counter >= MAX_STAGNATION:
            stagnation_counter = 0
            constructed_solution = remove_edges(data, best_sol.copy(), removed_percent=0.5)
            model_primary.setObjective(gp.quicksum((1-constructed_solution[e]) * model_primary._z[e] * data.G.edges[e]['c'] for e in data.G.edges), GRB.MINIMIZE)

            model_primary.optimize()
            if model_primary.status == GRB.INFEASIBLE:
                constructed_solution = best_sol.copy()
            else:
                z_solution = _z_solution(model_primary)
                bunch_size = initial_bunch_size

        while i < bunch_size:
            constructed_solution = append_embedding(constructed_solution, z_solution)
            non_zero_edges = [e for e in z_solution if z_solution[e] > 0.5]

            old_enhanced_constr = model_backup.getConstrByName('enhanced_constr')
            if old_enhanced_constr: model_backup.remove(old_enhanced_constr)
            model_backup.addConstr(gp.quicksum(model_backup._z[e] for e in data.G.edges if constructed_solution.get(e, 0) == 0) >= 1, name='enhanced_constr')

            model_backup.update()
            model_backup.optimize()
            if model_backup.status == GRB.INFEASIBLE:
                break
            z_solution = _z_solution(model_backup)
            i += 1
        
    return None