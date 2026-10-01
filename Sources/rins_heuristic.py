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
from optimization_models import check_instance_feasibility

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

def _solution_cost(data, sol):
    return sum(data.G.edges[e]['c'] for e, v in sol.items() if v > 0.5)

_feasibility_cache = {}

def _cached_feasibility(data, sol):
    key = frozenset(e for e, v in sol.items() if v > 0.5)
    if key not in _feasibility_cache:
        _feasibility_cache[key] = check_instance_feasibility(data, sol)

    return _feasibility_cache[key]


def greedy_remove_edges(data, sol, sort_mode='cost'):
    edges = list(data.G.edges(data=True))
    if sort_mode == 'cost':
        edges.sort(key=lambda e: e[2]['c'], reverse=True)
    elif sort_mode == 'random':
        std_random.shuffle(edges)
    elif sort_mode == 'cost_over_reliability':
        # Надёжность = 1 - вероятность отказа p
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

def rins_heuristic(data, updateCallback=None, verbose=True, dct=None, sync_event=None):
    while True:
        sol = dct['obj_vector'].copy()
        candidate_edges = [e for e in dct['incumbent'] if dct['incumbent'][e] == 1 and sol.get(e, 0) <= 1e-9]
        sorted_edges = sorted(candidate_edges, key=lambda e: data.G.edges[e]['c'])

        batch_size = max(1, len(sorted_edges) // 10)
        idx = 0
        while idx < len(sorted_edges):
            batch = sorted_edges[idx: idx + batch_size]
            for e in batch:
                sol[e] = 1
            if _cached_feasibility(data, sol):
                for e in batch:
                    sol[e] = 0
                for e in batch:
                    sol[e] = 1
                    if _cached_feasibility(data, sol):
                        break
                break
            else:
                idx += batch_size

        sol = greedy_remove_edges(data, sol, sort_mode='cost')
        sol = remove_dangling_chains(data, sol)
        cost = _solution_cost(data, sol)
        if verbose:
            print(f'RINS improved solution: cost = {cost}')
        if _cached_feasibility(data, sol) and cost < dct['UB']:
            updateUB(data.G, sol, updateCallback, True, sync_event, f'RINS new incumbent: {cost}')