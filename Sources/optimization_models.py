import math
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from common import models, probs, is_time_limit_exceeded
import copy
from itertools import product

def on_new_incumbent(model, where):
    if where == GRB.Callback.MIPSOL:
        model.terminate()


def prepare_main_model(G, appended_master_constrs=None):
    model_name = "Main"
    model = gp.Model(model_name)
    model.setParam("OutputFlag", 0)
    model.setParam("LogToConsole", 0)
    model.setParam("Threads", 3)
    model.setParam("MIPGap", 0.0)
    x_vars = gp.tupledict()
    for e in G.edges():
        x_vars[e] = model.addVar(obj=G.edges[e]["c"], vtype=GRB.BINARY, name=f"x_{e}")

    if appended_master_constrs:
        for i, cut in enumerate(appended_master_constrs):
            model.addConstr(gp.quicksum(x_vars[e] for e in cut) >= 1, name=f"mc_{i}")

    model._x = x_vars
    model.update()
    return model

def should_node_be_passed_common_model(G, node, source, dest_cluster, hubs, direction='forward'):
    if node == source:
        return True

    G_sub = G.subgraph([node, source] + dest_cluster + hubs)
    
    for v in dest_cluster:
        if direction == 'forward' and nx.has_path(G_sub,source,node) and nx.has_path(G_sub,node,v):
            return True
        # if direction == 'backward':
            # print(f'{node=}, {source=}, {v=}, {nx.has_path(G_sub,node,source)=}, {nx.has_path(G_sub,v,node)=}, {G_sub.edges=}')

        if direction == 'backward' and nx.has_path(G_sub,node,source) and nx.has_path(G_sub,v,node):
            return True

    return False


def should_node_be_passed(G, node, ordered_pair, clusters, hubs):
    G_sub = G.subgraph(clusters[ordered_pair[0]] + clusters[ordered_pair[1]] + hubs)
    
    nodes_i = clusters[ordered_pair[0]]
    nodes_j = clusters[ordered_pair[1]]

    for v1 in nodes_i:
        if node in nodes_j and nx.has_path(G_sub,v1,node):
            return True

        for v2 in nodes_j:
            if node in nodes_i and nx.has_path(G_sub,node,v2):
                return True

            if node in G_sub and v1 in G_sub and v2 in G_sub and nx.has_path(G_sub, v1, node) and nx.has_path(G_sub, node, v2):
                return True
    return False


####################################################################
###  Update X-dependent constriant
###  mdl - initial model
###  G - graph, x_sol - new solutio of the restricted master problem
####################################################################
def updateXconstraint(mdl, G, x_sol):
    old_constr_w = mdl.getConstrByName('orth_w')
    mdl.remove(old_constr_w)
    mdl.addConstr(gp.quicksum(x_sol[e] * mdl._vars_w[e] for e in G.edges if e in mdl._vars_w) <= 0, name='orth_w')
    mdl.update()
    return mdl

def is_edge_can_be_used_in_arc(G, clusters, arc, edge):
    for u in clusters[arc[0]]:
        for v in clusters[arc[1]]:
            if nx.has_path(G, u, edge[0]) and nx.has_path(G, edge[1], v):
                return True
    return False

def prepare_budget_model(
    data,
    ordered_pair,
    x_solution,
    is_backup = False,
):
    model_name = (f'Budget for {ordered_pair} , {is_backup=}')
    model = gp.Model(model_name)
    model.setParam("OutputFlag", 0)
    model.setParam("LogToConsole", 0)
    model.setParam("Threads", 3)
    model.setParam("MIPGap", 0.0)
    vars_w = gp.tupledict()
    vars_f = gp.tupledict()
    vars_pi = gp.tupledict()

    #Filter graph
    G_local = data.G.subgraph(data.clusters[ordered_pair[0]] + data.clusters[ordered_pair[1]] + data.hubs).copy()
    G_local.remove_edges_from([(u, v) for u in data.clusters[ordered_pair[1]] for v in G_local.successors(u)])
    G_local.remove_edges_from([(u, v) for v in data.clusters[ordered_pair[0]] for u in G_local.predecessors(v)])
    nodes_to_remove_out = [node for node, out_degree in G_local.out_degree() if out_degree == 0 and node in data.hubs]
    nodes_to_remove_in = [node for node, in_degree in G_local.in_degree() if in_degree == 0 and node in data.hubs]
    G_local = G_local.subgraph(set(G_local.nodes) - set(nodes_to_remove_out) - set(nodes_to_remove_in)).copy()

    i = ordered_pair[0]
    j = ordered_pair[1]

    clusters = data.clusters
    hubs = data.hubs
    budget = data.budget_backup if is_backup else data.budget_primary
    L = max(budget.values())

    delta = 0.01
    small_sigma = 1e-6

    # vars_w and vars_f initializing
    for e in G_local.edges():
        # vars_w[e] = model.addVar(obj=x_solution[e], vtype=GRB.BINARY, name=f"w_{e}")
        vars_w[e] = model.addVar(obj=1, vtype=GRB.BINARY, name=f"w_{e}")
        vars_f[e] = model.addVar(vtype=GRB.BINARY, name=f"f_{e}")

    # w_{u,v}+t_{u,v} <= 1
    model.addConstrs((vars_w[u, v] + vars_f[u, v] <= 1) for u, v in G_local.edges())
    # sum\limits_{e\in E}\hat x_e * w_e <= 0
    model.addConstr(gp.quicksum(x_solution[e] * vars_w[e] for e in G_local.edges() if e in x_solution) <= 0, name='orth_w')

    # potential variables
    for u in clusters[i]:
        for v in clusters[i] + clusters[j] + hubs:
            if v in clusters[i] and v != u:
                continue
            vars_pi[v,u] = model.addVar(name=f"pi_{v,u}")

    model.addConstrs((vars_pi[u,u] == 0) for u in clusters[i])
    model.addConstrs((vars_pi[v,u] - vars_pi[vv,u] - 1 - (L + 1) * (vars_w[vv, v] + vars_f[vv, v]) <= 0) 
            for u in clusters[i] for v in hubs + clusters[j] for vv in hubs + clusters[i] 
            if (vv,v) in G_local.edges and (vv,u) in vars_pi and (v,u) in vars_pi)
    model.addConstrs((vars_pi[v,u] >= L + 1) for u in clusters[i] for v in clusters[j])

    if is_backup:
        model.addConstr(gp.quicksum([math.log(e_attr["p"]) * vars_f[(e_out, e_in)] for e_out, e_in, e_attr in G_local.edges(data=True)]) - math.log(data.epsilon_backup) >= 0)
    else:
        model.addConstrs((vars_f[e] == 0) for e in G_local.edges())

    model._vars_w = vars_w
    model._vars_f = vars_f

    model.update()
    return model

def check_instance_feasibility(data, solution, ct=None):
    common_temp_models = []
    common_temp_models.append(prepare_common_model(data, solution, is_backup = False))
    common_temp_models.append(prepare_common_model(data, solution, is_backup = True))

    for m_obj in common_temp_models:
        updateXconstraint(m_obj, data.G, solution)
        m_obj.optimize(on_new_incumbent)
        if m_obj.status == GRB.INFEASIBLE:
            continue
        else:
            return False

    return True

def prepare_common_model(
    data,
    x_solution,
    is_backup=False
):
    """
    Separation model for ADDING edges (x_hat = 0).
    Finds a minimum set of edges W to add so that the budget/gluing
    constraints are violated.
    w[e] = 1  => edge e is added.
    """
    model = gp.Model("CommonSeparation")
    model.setParam("OutputFlag", 0)
    model.setParam("LogToConsole", 0)
    model.setParam("Threads", 3)

    G = data.G
    H = data.H
    clusters = data.clusters
    hubs = data.hubs
    start_node = data.start_node
    dest_node = data.dest_node
    budget = data.budget_backup if is_backup else data.budget_primary
    L = max(budget.values())

    M = (len(data.hubs) + 1) * (max(budget.values()) + 1)

    w = gp.tupledict()
    f = gp.tupledict()
    for e in G.edges:
        w[e] = model.addVar(obj=1, vtype=GRB.BINARY, name=f"w_{e}")
        f[e] = model.addVar(vtype=GRB.BINARY, name=f"t_{e}")
        model.addConstr(w[e] + f[e] <= 1)

    model.addConstr(gp.quicksum(x_solution.get(e, 0) * w[e] for e in G.edges) <= 0,
                    name="orth_w")

    alpha = gp.tupledict()
    for (i,j) in H.edges:
        for u in clusters[i]:
            for v in clusters[j]:
                alpha[u,v] = model.addVar(vtype=GRB.BINARY, name=f"alpha_{u,v}")


    pi = gp.tupledict()

    for (i, j) in H.edges:
        for u in clusters[i]:
            for v in hubs + clusters[i] + clusters[j]:
                if v in clusters[i] and u != v:
                    continue

                # if should_node_be_passed_common_model(G, v, u, clusters[j], hubs, direction='forward'):
                pi[v, u] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"pi_{v}_{u}")
            model.addConstr(pi[u, u] == 0)

    model.addConstrs((pi[v,u] <= pi[vv,u] + 1 + L * (w[vv,v] + f[vv,v])) 
        for (i,j) in H.edges for u in clusters[i] for v in clusters[j] + hubs for vv in clusters[i] + hubs if (vv,v) in G.edges and (v,u) in pi and (vv,u) in pi)

    model.addConstrs((pi[v,u] >= (L + 1)* alpha[u,v]) for (i,j) in H.edges for u in clusters[i] for v in clusters[j] if (v,u) in pi)
    model.addConstrs((pi[v,u] <= L + M * alpha[u,v]) for (i,j) in H.edges for u in clusters[i] for v in clusters[j] if (v,u) in pi)

    keys = list(clusters.keys())
    for comb in product(*clusters.values()):
        selected = dict(zip(keys, comb))
        model.addConstr(gp.quicksum(alpha[selected[i],selected[j]] for (i,j) in H.edges) >= 1)


    if is_backup:
        model.addConstr(
            gp.quicksum(math.log(G.edges[e]["p"]) * f[e] for e in G.edges)
            >= math.log(data.epsilon_backup),
            name="prob_backup"
        )
    else:
        for e in G.edges:
            model.addConstr(f[e] == 0)

    model._vars_w = w
    model._vars_f = f
    model.update()
    # model.write(f'toy_common.lp')
    # exit(0)
    return model

def prepare_multicommodity_model(data, mip_start = {}, is_backup = False):
    model_name = "Multicommodity model"
    model = gp.Model(model_name)
    model.setParam("OutputFlag", 0)
    model.setParam("LogToConsole", 0)
    model.setParam("Threads", 3)
    model.setParam("MIPGap", 0.0)
    vars_f = gp.tupledict()
    vars_y = gp.tupledict()
    vars_z = gp.tupledict()
    # vars_p = gp.tupledict()

    G = data.G
    H = data.H
    clusters = data.clusters
    hubs = data.hubs
    budgets = data.budget_backup if is_backup else data.budget_primary
    epsilon_backup = data.epsilon_backup

    sigma = 1e-6
    M = sum(G.edges[e]["b"] for e in G.edges)

    for (i, j) in H.edges:
        for (u, v) in G.edges:
            vars_f[(i, j), (u, v)] = model.addVar(
                vtype=GRB.BINARY, name=f"f_{(i,j)}_{(u,v)}"
            )

    for v in G.nodes:
        vars_y[v] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}")

    for (u, v) in G.edges:
        # vars_p[u, v] = model.addVar(obj=M, vtype=GRB.BINARY, name=f"p_{(u,v)}")
        # vars_z[u, v] = model.addVar(obj=G.edges[u,v]['c'] / (1.0 - G.edges[u,v]['p']), vtype=GRB.BINARY, name=f"z_{(u,v)}")
        vars_z[u, v] = model.addVar(obj=G.edges[u,v]['c'], vtype=GRB.BINARY, name=f"z_{(u,v)}")

    for (i, j) in H.edges:
        for u in clusters[i]:
            model.addConstr(
                gp.quicksum(
                    vars_f[(i, j), (u, v)]
                    for v in hubs + clusters[j]
                    if (u, v) in G.edges
                )
                - vars_y[u]
                == 0
            )

    for (i, j) in H.edges:
        for u in clusters[j]:
            model.addConstr(
                -gp.quicksum(
                    vars_f[(i, j), (v, u)]
                    for v in hubs + clusters[i]
                    if (v, u) in G.edges
                )
                + vars_y[u]
                == 0
            )

    model.addConstrs(
        (vars_f[(i, j), (u, v)] - vars_z[(u, v)] <= 0)
        for (i, j) in H.edges
        for (u, v) in G.edges
    )

    model.addConstrs((vars_z[u,v] - gp.quicksum(vars_f[(i,j),(u,v)] for (i,j) in data.H.edges) <= 0) for (u,v) in data.G.edges)

    model.addConstrs(
        (
            gp.quicksum(
                vars_f[(i, j), (u, v)]
                for u in hubs + clusters[i]
                if (u, v) in G.edges
            )
            - gp.quicksum(
                vars_f[(i, j), (v, u)]
                for u in hubs + clusters[j]
                if (v, u) in G.edges
            )
            == 0
        )
        for v in hubs
        for (i, j) in H.edges
    )

    model.addConstrs(
        (gp.quicksum(vars_y[v] for v in clusters[i]) == 1) for i in clusters
    )

    model.addConstrs(
        (
            gp.quicksum(
                vars_f[(i, j), (u, v)] * G.edges[u, v]["b"] for (u, v) in G.edges
            )
            - budgets[(i, j)]
            <= 0
        )
        for (i, j) in H.edges
    )


    model._z = vars_z
    model.update()
    return model

import math
import gurobipy as gp
from gurobipy import GRB

def prepare_local_embedding_model(
    data,
    x_solution,
    fixed_cluster,
    with_probs=False
):
    model = gp.Model(f"LocalEmbedding_{fixed_cluster}")
    model.setParam("OutputFlag", 0)
    model.setParam("LogToConsole", 0)
    model.setParam("MIPGap", 0.0)
    M = (len(data.hubs) + 1) * (max(data.budget_backup.values()) + 1)

    G = data.G
    H = data.H
    clusters = data.clusters
    hubs = data.hubs
    j = fixed_cluster

    w = gp.tupledict()
    f = gp.tupledict()
    for e in G.edges:
        w[e] = model.addVar(vtype=GRB.BINARY, name=f"w_{e}")
        f[e] = model.addVar(vtype=GRB.BINARY, name=f"f_{e}")
        model.addConstr(w[e] + f[e] <= 1)

    lambda_fwd = gp.tupledict()
    lambda_bwd = gp.tupledict()
    predecessors = list(H.predecessors(fixed_cluster))
    successors = list(H.successors(fixed_cluster))

    for i in predecessors:
        for v in hubs + clusters[i] + clusters[j]:
            lambda_fwd[(i, j), v] = model.addVar(lb=0.0, ub=1.0,
                                                 vtype=GRB.CONTINUOUS,
                                                 name=f"lfwd_{i}_{j}_{v}")
    for k in successors:
        for v in hubs + clusters[j] + clusters[k]:
            lambda_bwd[(j, k), v] = model.addVar(lb=0.0, ub=1.0,
                                                 vtype=GRB.CONTINUOUS,
                                                 name=f"lbwd_{j}_{k}_{v}")

    tau_fwd = gp.tupledict()
    tau_bwd = gp.tupledict()
    for v in clusters[j]:
        tau_fwd[v] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                  name=f"tauf_{v}")
        tau_bwd[v] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                  name=f"taub_{v}")


    model.setObjective(gp.quicksum(w[e] for e in G.edges), GRB.MINIMIZE)

    model.addConstr(gp.quicksum(x_solution.get(e, 0) * w[e] for e in G.edges) == 0,
                    name="orth_w")

    for i in predecessors:
        for v in clusters[i]:
            if ((i, j), v) in lambda_fwd:
                model.addConstr(lambda_fwd[(i, j), v] == 0, name=f"lfwd_src_{i}_{j}_{v}")

    for i in predecessors:
        for v in clusters[j] + hubs:
            if ((i, j), v) not in lambda_fwd:
                continue

            max_terms = []
            for u in hubs + clusters[i]:
                if not G.has_edge(u, v):
                    continue
                max_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                       name=f"max_fwd_{i}_{j}_{u}_{v}")
                model.addGenConstrMax(max_var,
                                      [lambda_fwd[(i, j), u], w[(u, v)], f[(u, v)]])
                max_terms.append(max_var)

            if max_terms:
                min_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                       name=f"min_fwd_{i}_{j}_{v}")
                model.addGenConstrMin(min_var, max_terms)
                model.addConstr(lambda_fwd[(i, j), v] == min_var)
            else:
                model.addConstr(lambda_fwd[(i, j), v] == 1)


    for k in successors:
        for v in clusters[k]:
            if ((j, k), v) in lambda_bwd:
                model.addConstr(lambda_bwd[(j, k), v] == 0, name=f"lbwd_dst_{j}_{k}_{v}")


    for k in successors:
        for v in clusters[j] + hubs:
            if ((j, k), v) not in lambda_bwd:
                continue
            max_terms = []
            for u in hubs + clusters[k]:
                if not G.has_edge(v, u):
                    continue
                max_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                       name=f"max_bwd_{j}_{k}_{v}_{u}")
                model.addGenConstrMax(max_var,
                                      [lambda_bwd[(j, k), u], w[(v, u)], f[(v, u)]])
                max_terms.append(max_var)

            if max_terms:
                min_var = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                       name=f"min_bwd_{j}_{k}_{v}")
                model.addGenConstrMin(min_var, max_terms)
                model.addConstr(lambda_bwd[(j, k), v] == min_var)
            else:
                model.addConstr(lambda_bwd[(j, k), v] == 1)


    for v in clusters[j]:
        list_for_max = [lambda_fwd[(i, j), v] for i in predecessors
                        if ((i, j), v) in lambda_fwd]
        if list_for_max:
            model.addGenConstrMax(tau_fwd[v], list_for_max)
        else:
            model.addConstr(tau_fwd[v] == 1)


    for v in clusters[j]:
        list_for_max = [lambda_bwd[(j, k), v] for k in successors
                        if ((j, k), v) in lambda_bwd]
        if list_for_max:
            model.addGenConstrMax(tau_bwd[v], list_for_max)
        else:
            model.addConstr(tau_bwd[v] == 1)


    min_terms = []
    max_tau = gp.tupledict()
    for v in clusters[j]:
        max_tau[v] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                               name=f"max_tau_{v}")
        model.addGenConstrMax(max_tau[v], [tau_fwd[v], tau_bwd[v]])
        min_terms.append(max_tau[v])

    if min_terms:
        min_overall = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                   name="min_overall")
        model.addGenConstrMin(min_overall, min_terms)
        model.addConstr(min_overall >= 1, name="gluing_local")


    if with_probs:
        model.addConstr(gp.quicksum(math.log(G.edges[e]["p"]) * f[e] for e in G.edges)
                        >= math.log(data.epsilon_backup), name="prob")
    else:
        for e in G.edges:
            model.addConstr(f[e] == 0)

    model._vars_w = w
    model._vars_f = f
    model._lambda_fwd = lambda_fwd
    model._lambda_bwd = lambda_bwd
    model._tau_fwd = tau_fwd
    model._tau_bwd = tau_bwd

    model.update()
    return model


def prepare_local_common_model(
    data,
    x_solution,
    fixed_cluster,
    is_backup=False
):
    model = gp.Model("CommonSeparation")
    model.setParam("OutputFlag", 0)
    model.setParam("LogToConsole", 0)
    model.setParam("Threads", 3)
    model.setParam("MIPGap", 0.0)

    G = data.G
    H = data.H
    clusters = data.clusters
    hubs = data.hubs
    start_node = data.start_node
    dest_node = data.dest_node
    budget = data.budget_backup if is_backup else data.budget_primary
    j = fixed_cluster
    predecessors = list(H.predecessors(j))
    successors = list(H.successors(j))
    L = max(budget.values())

    M = (len(data.hubs) + 1) * (max(budget.values()) + 1)

    vars_w = gp.tupledict()
    vars_f = gp.tupledict()
    for e in G.edges:
        vars_w[e] = model.addVar(obj=1, vtype=GRB.BINARY, name=f"w_{e}")
        vars_f[e] = model.addVar(vtype=GRB.BINARY, name=f"f_{e}")
        model.addConstr(vars_w[e] + vars_f[e] <= 1)

    model.addConstr(gp.quicksum(x_solution.get(e, 0) * vars_w[e] for e in G.edges) <= 0,
                    name="orth_w")

    ftau = gp.tupledict()
    btau = gp.tupledict()

    min_ftau = gp.tupledict()
    min_btau = gp.tupledict()

    max_tau = gp.tupledict()

    for u in clusters[j]:
        ftau[u] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"ftau_{u}")
        btau[u] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"btau_{u}")


    pi = gp.tupledict()
    alpha = gp.tupledict()

    for i in predecessors:
        for u in clusters[i]:
            for v in hubs + clusters[i] + clusters[j]:
                pi[v, u] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"pi_{i}_{j}_{v}_{u}")
                if v in clusters[j]:
                    alpha[v, u] = model.addVar(vtype=GRB.BINARY, name=f"alpha_{i}_{j}_{v}_{u}")
            model.addConstr(pi[u, u] == 0)

    for k in successors:
        for u in clusters[k]:
            for v in hubs + clusters[j] + clusters[k]:
                pi[v, u] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"pi_{j}_{k}_{v}_{u}")
                if v in clusters[j]:
                    alpha[u, v] = model.addVar(vtype=GRB.BINARY, name=f"alpha_{j}_{k}_{v}_{u}")
            model.addConstr(pi[u, u] == 0)

    #############################################################################
    # Forward potential propagation
    model.addConstrs((pi[v,u] <= pi[vv,u] + 1 + L * (vars_w[vv,v]+vars_f[vv,v])) 
    for i in predecessors for u in clusters[i]
    for v in clusters[j] + hubs for vv in clusters[i] + hubs if (vv,v) in G.edges)
        
    model.addConstrs((pi[v,u] - (L+1) * alpha[v,u] >= 0) 
        for i in predecessors for u in clusters[i] for v in clusters[j])

    model.addConstrs((pi[v,u] - L - M * alpha[v,u] <= 0)
        for i in predecessors for u in clusters[i] for v in clusters[j])

    # Backward potential propagation
    model.addConstrs((pi[v,u] <= pi[vv,u] + 1 
        + L * (vars_w[v,vv]+vars_f[v,vv])) 
    for k in successors for u in clusters[k]
    for v in clusters[j] + hubs for vv in clusters[k] + hubs 
    if (v,vv) in G.edges)
        
    model.addConstrs((pi[u,v] - (L+1) * alpha[v,u] >= 0) 
        for k in successors for v in clusters[k] for u in clusters[j])

    model.addConstrs((pi[u,v] - L - M * alpha[v,u] <= 0)
        for k in successors for v in clusters[k] for u in clusters[j])

    # Tau constraints
    
    for v in clusters[j]:
        for i in predecessors:
            min_ftau[i] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                           name=f"min_ftau_{i}")
            model.addGenConstrMin(min_ftau[i], [alpha[v,u] for u in clusters[i]])
        
        model.addGenConstrMax(ftau[v], [min_ftau[i] for i in predecessors])

        for k in successors:
            min_btau[k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                           name=f"min_btau_{k}")
            model.addGenConstrMin(min_btau[k], [alpha[u,v] for u in clusters[k]])
        
        model.addGenConstrMax(btau[v], [min_btau[k] for k in successors])
    
    # Minimax constraint: THE CUT CONSTRAINT
    cut_var = model.addVar(vtype=GRB.BINARY,name=f"cut_var")
    for u in clusters[j]:
        max_tau[u] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,name=f"max_tau[u]")
        model.addGenConstrMax(max_tau[u], [ftau[u], btau[u]])

    model.addGenConstrMin(cut_var, [max_tau[u] for u in clusters[j]])
    model.addConstr(cut_var >= 1, name=f"cut_constr")

    # ----- Probabilistic constraints -----
    if is_backup:
        model.addConstr(
            gp.quicksum(math.log(G.edges[e]["p"]) * vars_f[e] for e in G.edges)
            >= math.log(data.epsilon_backup),
            name="prob_backup"
        )
    else:
        for e in G.edges:
            model.addConstr(vars_f[e] == 0)

    model._vars_w = vars_w
    model._vars_f = vars_f
    model.update()
    return model