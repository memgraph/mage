import sys
import numpy as np
from distance_module.distance_util import calculate_distance_between_points

try:
    import networkx as nx
except ImportError as import_error:
    sys.stderr.write((
        '\n'
        'NOTE: Please install networkx to be able to use graph_analyzer '
        'module. Using Python:\n'
        + sys.version +
        '\n'))
    raise import_error

DEFAULT_SOLVING_METHOD = 'greedy'

def create_distance_matrix(points):
    '''
    Creates a quadratic matrix of distances between points.
    :param points: List of dictionaries with lat and lng coordinates
    :return: Distance matrix
    '''

    dm = np.zeros([len(points), len(points)])

    for i in range(0, len(points) - 1):
        for j in range(i + 1, len(points)):
            d = calculate_distance_between_points(points[i], points[j])
            dm[i][j] = dm[j][i] = d

    return dm

def solve_2_approx(dm):
    '''
    Solves the tsp_module problem with 2-approximation.
    :param dm: Distance matrix.
    :return: List of indices - path between them (based on distance matrix indexes)
    '''

    mst = get_mst(dm)
    path = [x for x in nx.dfs_preorder_nodes(mst)]
    path.append(path[0])

    return path

def solve_1_5_approx(dm):
    '''
    Solves the tsp_module problem with 1.5-approximation (Christofides algorithm).
    :param distance_matrix: Distance matrix.
    :return: List of indices - path between them (based on distance matrix indexes)
    '''

    mst = get_mst(dm)
    odd_matchings = [x[0] for x in list(filter(lambda x: x[1] % 2 == 1, list(mst.degree)))]
    matches = get_perfect_matchings(odd_matchings)

    all_edges = list(mst.edges)
    all_edges.extend(matches)

    euler_circuit = get_euler_circuit(all_edges)
    path = get_hamiltonian_circuit(euler_circuit)

    return path

def solve_greedy(dm):
    '''
    Solves the tsp_module problem with greedy method of taking the closest node to the last.
    :param distance_matrix: Distance matrix.
    :return: List of indices - path between them (based on distance matrix indexes)
    '''

    path = []
    visited_vert = dict()
    path.append(0)
    visited_vert[0] = True

    while len(path) != len(dm):
        last = path[-1]
        min_index, min_val = -1, -1

        for i in range(len(dm)):
            value = dm[last][i]
            if last != i and (min_index == -1 or min_val > value) and i not in visited_vert.keys():
                min_index = i
                min_val = value

        path.append(min_index)
        visited_vert[min_index] = True

    path.append(0)

    return path

tsp_solving_methods = {
    '2_approx':solve_2_approx,
    '1.5_approx':solve_1_5_approx,
    DEFAULT_SOLVING_METHOD:solve_greedy
}

def get_hamiltonian_circuit(euler_circuit):
    '''
    Deletes duplicates of the Euler circuit in order to form hamiltonian circuit where no vertex is
    visited twice or more times
    :param euler_circuit: Eulerian path
    :return:
    '''

    path = []
    [path.append(x[0]) for x in euler_circuit]
    path = list(dict.fromkeys(path))
    path.append(path[0])

    return path

def get_euler_circuit(tum_edges):
    '''
    Uses nx library for finding an Eulerian circuit
    :param tum_edges: Union of mst and matchings edges
    :return: Eulerian path generator
    '''

    g = nx.MultiGraph()

    for edge in tum_edges:
        g.add_edge(edge[0], edge[1])

    path = nx.eulerian_path(g, source=tum_edges[0][0])

    return path

def get_perfect_matchings(odd_matchings):
    '''
    Dummy perfect matchings method which takes every 2 vertexes and combines them to an edge
    #TODO, real perfect matchings with minimum cost
    :param odd_matchings: List of vertexes with odd degree
    :return: List of matched edges
    '''

    matched_edges = [(odd_matchings[i], odd_matchings[i+1]) for i in range(0, len(odd_matchings), 2)]

    return matched_edges

def get_mst(dm):
    '''
    Creates the minimum spanning tree using nx.
    :param dm: Distance matrix
    :return: Minimum spanning tree
    '''

    g = nx.Graph()

    for i in range(len(dm) - 1):
        for j in range(i + 1, len(dm)):
            g.add_edge(i, j, weight=dm[i][j])

    mst = nx.minimum_spanning_tree(g)

    return mst