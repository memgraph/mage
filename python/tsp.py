import mgp
import sys
import numpy as np
import math
from typing import List

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


DEFAULT_SOLVING_METHOD = '1.5_approx'
KM_MULTIPLIER = 0.001
LATITUDE = 'lat'
LONGITUDE = 'lng'
VALID_METRICS = ['m', 'km']

def calculate_distance_between_points(start, end, metrics='m'):
    '''
    Returns distance based on the metrics between 2 points.
    :param start: Start node - dictionary with lat and lng
    :param end: End node - dictionary with lat and lng
    :param metrics: m - in metres, km - in kilometres
    :return: float distance
    '''

    if (LATITUDE not in start.keys()
        or LONGITUDE not in start.keys()
        or LATITUDE not in end.keys()
        or LONGITUDE not in end.keys()):
        return None

    if not isinstance(metrics, str) or metrics.lower() not in VALID_METRICS:
        return None

    lat_1 = start[LATITUDE]
    lng_1 = start[LONGITUDE]
    lat_2 = end[LATITUDE]
    lng_2 = end[LONGITUDE]

    R = 6371E3
    pi_radians = math.pi / 180.00

    phi_1 = lat_1 * pi_radians
    phi_2 = lat_2 * pi_radians
    delta_phi = (lat_2 - lat_1) * pi_radians
    delta_lambda = (lng_2 - lng_1) * pi_radians

    sin_delta_phi = math.sin(delta_phi / 2.0)
    sin_delta_lambda = math.sin(delta_lambda / 2.0)

    a = sin_delta_phi ** 2 + math.cos(phi_1) * math.cos(phi_2) * (sin_delta_lambda ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    #Distance in metres
    distance = R * c

    if metrics.lower() == 'km':
        distance *= KM_MULTIPLIER

    return distance


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
            if d is None:
                return None
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
    odd_matchings = [x[0] for x in filter(lambda x: x[1] % 2 == 1, mst.degree)]
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
    'greedy':solve_greedy,
    DEFAULT_SOLVING_METHOD:solve_1_5_approx,
}


@mgp.read_proc
def solve(context:mgp.ProcCtx,
          points:List[mgp.Vertex],
          method:str = DEFAULT_SOLVING_METHOD
          ) -> mgp.Record(sources=List[mgp.Vertex],
                          destinations=List[mgp.Vertex]):
    '''
    The tsp_module solver returns 2 fields whose elements at indexes are correlated

      * `sources` - elements from 1st to n-1th element
      * `destinations` - elements from 2nd to nth element

    The pairs of them represent individual edges between 2 nodes in the graph.

    The required argument is the list of cities one wants to find the path from.
    The optional argument `method` is by default 'greedy'. Other arguments that can be
    specified are '2-approx' and '1.5-approx'

    The procedure can be invoked in openCypher using the following calls:
    MATCH (n:Point)
    WITH collect(n) as points
    CALL tsp_module.solve(points) YIELD sources, destinations;
    '''

    if not all(isinstance(x, mgp.Vertex) for x in points):
        return mgp.Record(sources=None, destinations=None)

    dm = create_distance_matrix(
        [dict(x.properties.items()) for x in points]
    )

    if dm is None:
        return mgp.Record(sources=None, destinations=None)

    if method.lower() not in tsp_solving_methods.keys():
        method = DEFAULT_SOLVING_METHOD

    order = tsp_solving_methods[method](dm)

    sources = [points[order[x]] for x in range(len(points) - 1)]
    destinations = [points[order[x]] for x in range(1, len(points))]

    return mgp.Record(sources=sources, destinations=destinations)


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
