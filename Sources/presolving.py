import networkx as nx
import copy

def filter_graph(G, H, clusters, hubs, budget_backup):
    edges_to_remove = [(e1,e2) for (e1,e2) in G.edges]
    for (e1,e2) in G.edges:
        to_removed = True
        for (i,j) in H.edges:
            for u in clusters[i]:
                for v in clusters[j]:
                    try:
                        G_local = G.subgraph(set(G.nodes)-set([node for nodes in clusters.values() for node in nodes]) | {u,v,e1,e2})
                        length_sub_path_1 = nx.shortest_path_length(G_local, source=u, target=e1, weight='b')
                        length_sub_path_2 = nx.shortest_path_length(G_local, source=e2, target=v, weight='b')
                        if length_sub_path_1 + length_sub_path_2 + 1 <= budget_backup:
                            edges_to_remove.remove((e1,e2))
                            to_removed = False
                            break
                    except nx.NetworkXNoPath:
                        continue
                if not to_removed: break
            if not to_removed: break
    
    G.remove_edges_from(edges_to_remove)
    return G


###################################
### Task presolving
### Input:  data (for  G, H, clusters, hubs)
### Returns: presolved G (same to data.G), incumbent, UB - if the instance is feasible
###          None - otherwise   
###################################
def presolve_a_task(data, constraint_types):
   # filter graph - remove all nodes that cannot be used in some route
    G = data.G
    H = data.H 
    clusters = data.clusters
    hubs = data.hubs

    if 'budget' in constraint_types or 'local_common' in constraint_types or 'common' in constraint_types:
        print(f"Before filtering: {len(G.edges)=}")
        G = filter_graph(G, H, clusters, hubs, max(data.budget_backup.values()))
        data.G = G
        print(f"After filtering: {len(G.edges)=}")
    
    incumbent = {e : 1 for e in data.G.edges}

    UB = sum([attr["c"] for e1,e2,attr in data.G.edges(data=True)])
    
    return G, incumbent, UB

