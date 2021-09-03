from enum import Enum

import networkx as nx


# TODO: implement for directed graphs (+ change division with 2.0 to 1.0)
# TODO: add partial BFS optimization

class Operation(Enum):
    INSERTION = "insertion"
    DELETION = "deletion"


def icentral(G: nx.Graph, from_: int, to_: int, bc: dict, operation: str) -> dict:
    # H will be the graph changes will be made on
    H = G.copy()

    biconnected_components: list
    biconnected_components_edges: list

    op = Operation(operation)

    if op == Operation.INSERTION:
        H.add_edge(from_, to_)

        # get list of biconnected components
        # this has to be done AFTER edge INSERTION (in case two biconnected components merge into one)
        biconnected_components = list(nx.biconnected_components(H))

        # these edges are needed to determine which biconnected component is affected
        # in other words - to find which biconnected component the inserted edge belongs to
        biconnected_components_edges = list(nx.biconnected_component_edges(H))

    elif op == Operation.DELETION:
        H.remove_edge(from_, to_)

        # get list of biconnected components in a list
        # this has to be done BEFORE edge DELETION (in case one biconnected components decomposes into two)
        biconnected_components = list(nx.biconnected_components(G))

        # these edges are needed to determine which biconnected component is affected
        # in other words - to find which biconnected component the inserted edge belongs to
        biconnected_components_edges = list(nx.biconnected_component_edges(G))

    # find index of affected component
    index_of_affected_biconnected_component: int
    for i, edges_of_component in enumerate(biconnected_components_edges):
        if (from_, to_) in edges_of_component or (to_, from_) in edges_of_component:
            index_of_affected_biconnected_component = i
            break

    if op == Operation.INSERTION:
        # get subgraph of affected biconnected component without added edge (initial graph)
        Be_subgraph = G.subgraph(biconnected_components[index_of_affected_biconnected_component])

        # get subgraph of affected biconnected component with added edge (changed graph)
        Be_subgraph_changed = H.subgraph(biconnected_components[index_of_affected_biconnected_component])

    elif op == Operation.DELETION:

        # get subgraph of affected biconnected component with edge (initial graph)
        Be_subgraph = G.subgraph(biconnected_components[index_of_affected_biconnected_component])

        # get subgraph of affected biconnected component without edge (changed graph)
        Be_subgraph_changed = H.subgraph(biconnected_components[index_of_affected_biconnected_component])

    # TODO: not sure how this part would work in directed graphs
    # we make two BFS' within the affected component - one starting from from_ node, the other starting from to_ node
    # we keep the distances from the source node in the two dictionaries below

    distances1 = bfs_counting_hops(Be_subgraph, from_)
    distances2 = bfs_counting_hops(Be_subgraph, to_)

    # we create the set q
    q = set()
    for node in distances1.keys():
        if distances1[node] != distances2[node]:
            q.add(node)

    # get articulation points
    articulation_points: list

    if op == Operation.INSERTION:
        articulation_points = list(nx.algorithms.components.articulation_points(H))
    elif op == Operation.DELETION:
        articulation_points = list(nx.algorithms.components.articulation_points(G))

    # PART 1 (lines 11 - 25 of pseudocode)
    for s in q:
        # create sigma_s and predecessors_s
        sigma_s = dict()
        predecessors_s = dict()

        # sigma_s - the starting node is s and the target nodes (t) are all nodes IN Be
        # predecessors_s - predecessors of a node on the shortest path between s (source) and current node (t)
        # we want to count how many shortest paths are between s and t
        for t in Be_subgraph.nodes:
            sigma_s[t] = len(list(nx.algorithms.shortest_paths.generic.all_shortest_paths(G, source=s, target=t)))

            # TODO: extend for directed graphs
            predecessors_s[t] = nx.algorithms.shortest_paths.unweighted.predecessor(G, source=s, target=t)

        sigma_s[s] = 0

        # initialize delta_s(v) and delta_Gs(v)
        delta_s = dict()
        delta_Gs = dict()

        for v in Be_subgraph.nodes:
            delta_Gs[v] = 0
            delta_s[v] = 0

        # get nodes in reverse BFS order from source node s
        nodes_in_reverse_bfs_order = get_reverse_BFS_order(Be_subgraph, s)

        for w in nodes_in_reverse_bfs_order:
            if s in articulation_points and w in articulation_points:
                delta_Gs[w] = get_cardinality_of_Gi(G, s,
                                                    biconnected_components[index_of_affected_biconnected_component]) * \
                              get_cardinality_of_Gi(G, w,
                                                    biconnected_components[index_of_affected_biconnected_component])

            for p in predecessors_s[w]:
                delta_s[p] = delta_s[p] + (float(sigma_s[p]) / float(sigma_s[w])) * (1 + delta_s[w])

                if s in articulation_points:
                    delta_Gs[p] = delta_Gs[p] + delta_Gs[w] * (float(sigma_s[p]) / float(sigma_s[w]))

            if w != s:
                bc[w] = bc[w] - float(delta_s[w]) / 2.0

            if s in articulation_points:
                bc[w] = bc[w] - delta_s[w] * get_cardinality_of_Gi(G, s,
                                                                   biconnected_components[
                                                                       index_of_affected_biconnected_component])
                bc[w] = bc[w] - float(delta_Gs[w]) / 2.0

        # PART 2 (lines 26 - 40 of pseudocode)

        # create sigma_s2 and predecessors_s2 (exactly the same as in the first part of the algorithm - the only
        # difference is the added/deleted edge)
        sigma_s2 = dict()
        predecessors_s2 = dict()

        # sigma_s2 - the starting node is s and the target nodes (t) are all nodes IN Be' (Be' == Be for vertices)
        # predecessors_s2 - predecessors of a node on shortest path between s (source) and current node (t)
        # we want to count how many shortest paths are between s and t
        for t in Be_subgraph_changed.nodes:
            sigma_s2[t] = len(list(nx.algorithms.shortest_paths.generic.all_shortest_paths(H,
                                                                                           source=s,
                                                                                           target=t)))

            # TODO: extend for directed graphs
            predecessors_s2[t] = nx.algorithms.shortest_paths.unweighted.predecessor(H,
                                                                                     source=s,
                                                                                     target=t)

        sigma_s2[s] = 0

        # initialize delta_s2(v) and delta_Gs2(v)
        delta_s2 = dict()
        delta_Gs2 = dict()

        for v in Be_subgraph_changed.nodes:
            delta_Gs2[v] = 0
            delta_s2[v] = 0

        # get nodes in reverse BFS order from source node s
        nodes_in_reverse_bfs_order = get_reverse_BFS_order(Be_subgraph_changed, s)

        for w in nodes_in_reverse_bfs_order:
            if s in articulation_points and w in articulation_points:
                delta_Gs2[w] = get_cardinality_of_Gi(H, s,
                                                     biconnected_components[
                                                         index_of_affected_biconnected_component]) * \
                               get_cardinality_of_Gi(H, w,
                                                     biconnected_components[
                                                         index_of_affected_biconnected_component])

            for p in predecessors_s2[w]:
                delta_s2[p] = delta_s2[p] + (float(sigma_s2[p]) / float(sigma_s2[w])) * (1 + delta_s2[w])

                if s in articulation_points:
                    delta_Gs2[p] = delta_Gs2[p] + delta_Gs2[w] * (float(sigma_s2[p]) / float(sigma_s2[w]))

            if w != s:
                bc[w] = bc[w] + float(delta_s2[w]) / 2.0

            if s in articulation_points:
                bc[w] = bc[w] + delta_s2[w] * get_cardinality_of_Gi(H, s,
                                                                    biconnected_components[
                                                                        index_of_affected_biconnected_component])
                bc[w] = bc[w] + float(delta_Gs2[w]) / 2.0

    return bc


# get number of nodes in subgraph connected to Be via articulation point a
def get_cardinality_of_Gi(G: nx.Graph, a: int, V_Be: list) -> int:
    visited = list()
    q = list()

    count = 0

    visited.append(a)
    q.append(a)

    while q:
        a = q.pop(0)
        for n in G.neighbors(a):
            if n not in V_Be:
                if n not in visited:
                    visited.append(n)
                    q.append(n)
                    count += 1

    return count


# function for getting reverse BFS order
def get_reverse_BFS_order(subgraph: nx.Graph, s: int) -> list:
    visited = list()
    q = list()

    visited.append(s)
    q.append(s)

    while q:
        s = q.pop(0)
        for n in subgraph.neighbors(s):
            if n not in visited:
                visited.append(n)
                q.append(n)

    visited.reverse()

    return visited


# version of BFS counting and returning number of hops from node to source node
def bfs_counting_hops(G: nx.Graph, s: int) -> dict:
    distances = dict()

    distances[s] = 0

    visited = list()
    q = list()

    visited.append(s)
    q.append((s, 0))

    while q:
        s, distance = q.pop(0)

        for n in G.neighbors(s):

            if n not in visited:
                visited.append(n)
                q.append((n, distance + 1))
                distances[n] = distance + 1

    return distances


# example from paper
def make_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(1, 21))
    G.add_edge(9, 10)
    G.add_edge(9, 11)
    G.add_edge(10, 1)
    G.add_edge(11, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 4)
    G.add_edge(2, 3)
    G.add_edge(4, 3)
    G.add_edge(4, 6)
    G.add_edge(6, 7)
    G.add_edge(7, 8)
    G.add_edge(6, 12)
    G.add_edge(6, 14)
    G.add_edge(12, 13)
    G.add_edge(14, 13)
    G.add_edge(3, 5)
    G.add_edge(8, 5)
    G.add_edge(5, 15)
    G.add_edge(5, 19)
    G.add_edge(19, 17)
    G.add_edge(15, 17)
    G.add_edge(17, 16)
    G.add_edge(17, 20)
    G.add_edge(16, 18)
    G.add_edge(20, 18)

    return G


# example from paper
def make_graph2() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(1, 21))
    G.add_edge(9, 10)
    G.add_edge(9, 11)
    G.add_edge(10, 1)
    G.add_edge(11, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 4)
    G.add_edge(2, 3)
    G.add_edge(4, 3)
    G.add_edge(4, 6)
    G.add_edge(6, 7)
    G.add_edge(7, 8)
    G.add_edge(6, 12)
    G.add_edge(6, 14)
    G.add_edge(12, 13)
    G.add_edge(14, 13)
    G.add_edge(3, 5)
    G.add_edge(8, 5)
    G.add_edge(5, 15)
    G.add_edge(5, 19)
    G.add_edge(19, 17)
    G.add_edge(15, 17)
    G.add_edge(17, 16)
    G.add_edge(17, 20)
    G.add_edge(16, 18)
    G.add_edge(20, 18)
    G.add_edge(4, 8)

    return G


def make_graph_smaller_example() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(1, 8))
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(1, 3)
    G.add_edge(2, 4)
    G.add_edge(4, 5)
    G.add_edge(5, 6)
    G.add_edge(6, 7)
    G.add_edge(7, 4)

    return G


def main():
    # G = make_graph()
    #
    # G_added_edge = make_graph()
    # G_added_edge.add_edge(4, 8)
    #
    # bc_brandes_added_edge = nx.betweenness_centrality(G_added_edge, normalized=False)
    # bc_incentral_added_edge = icentral(G, 4, 8, nx.betweenness_centrality(G, normalized=False), "insertion")
    #
    # for k in bc_brandes_added_edge.keys():
    #     print(f"{k}: {bc_incentral_added_edge[k]} <-> {bc_brandes_added_edge[k]}")

    H = make_graph2()
    H_deleted_edge = make_graph2()
    H_deleted_edge.remove_edge(4, 6)
    bc_brandes_deleted_edge = nx.betweenness_centrality(H_deleted_edge, normalized=False)
    bc_icentral_deleted_edge = icentral(H, 4, 6,
                                        nx.betweenness_centrality(H, normalized=False), "deletion")

    for k in bc_icentral_deleted_edge.keys():
        print(f"{k}: {bc_icentral_deleted_edge[k]} <-> {bc_brandes_deleted_edge[k]}")


def example():
    H = make_graph_smaller_example()

    bc = nx.betweenness_centrality(H, normalized=False)
    bc_incremental = icentral(H, 4, 6, bc.copy(), "insertion")
    H.add_edge(4, 6)
    bc_brandes = nx.betweenness_centrality(H, normalized=False)

    for k in bc.keys():
        print(f"{k}: {bc[k]} -> {bc_brandes[k]} <-> {bc_incremental[k]}")


if __name__ == '__main__':
    main()
    # example()