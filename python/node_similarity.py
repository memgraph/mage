from math import sqrt
from typing import Set, List, Callable
from itertools import product

import mgp


# not implemented for directed graphs
# not implemented for multigraphs
# not taking into account edge weights

@mgp.read_proc
def jaccard(
        context: mgp.ProcCtx,
        node1: mgp.Any,
        node2: mgp.Any,
        mode: str = 'c'
) -> mgp.Record(node1=mgp.Vertex, node2=mgp.Vertex, similarity=mgp.Number):
    """
    This procedure calls the method for calculating the Jaccard similarity between two nodes (or two lists of nodes)
    and returns 3 fields

    :return node1: The first node
    :return node2: The second node
    :return similarity: The Jaccard similarity of the first and the second node

    The input arguments consist of

    :param mgp.Vertex or Tuple[mgp.Vertex] node1: A node or a tuple of nodes
    :param mgp.Vertex or Tuple[mgp.Vertex] node2: A node or a tuple od nodes
    :param str mode: Can be `p` for pairwise similarity calculation or `c` for calculating the similarity of the
    Cartesian product of given tuples. Default value is `c`.
    """

    nodes1: tuple
    nodes2: tuple

    if isinstance(node1, mgp.Vertex):
        nodes1 = tuple([node1])
    elif isinstance(node1, tuple):
        nodes1 = node1
    else:
        raise TypeError("Invalid type of first argument.")

    if isinstance(node2, mgp.Vertex):
        nodes2 = tuple([node2])
    elif isinstance(node2, tuple):
        nodes2 = node2
    else:
        raise TypeError("Invalid type of second argument.")

    result = _calculate_similarity(nodes1, nodes2, _calculate_jaccard, mode)

    return [mgp.Record(node1=n1, node2=n2, similarity=similarity) for n1, n2, similarity in result]


@mgp.read_proc
def overlap(
        context: mgp.ProcCtx,
        node1: mgp.Any,
        node2: mgp.Any,
        mode: str = 'c'
) -> mgp.Record(node1=mgp.Vertex, node2=mgp.Vertex, similarity=mgp.Number):
    """
    This procedure calls the method for calculating the overlap similarity between two nodes (or two lists of nodes)
    and returns 3 fields

    :return node1: The first node
    :return node2: The second node
    :return similarity: The overlap similarity of the first and the second node

    The input arguments consist of

    :param mgp.Vertex or Tuple[mgp.Vertex] node1: A node or a tuple of nodes
    :param mgp.Vertex or Tuple[mgp.Vertex] node2: A node or a tuple od nodes
    :param str mode: Can be `p` for pairwise similarity calculation or `c` for calculating the similarity of the
    Cartesian product of given tuples. Default value is `c`.
    """

    nodes1: tuple
    nodes2: tuple

    if isinstance(node1, mgp.Vertex):
        nodes1 = tuple([node1])
    elif isinstance(node1, tuple):
        nodes1 = node1
    else:
        raise TypeError("Invalid type of first argument.")

    if isinstance(node2, mgp.Vertex):
        nodes2 = tuple([node2])
    elif isinstance(node2, tuple):
        nodes2 = node2
    else:
        raise TypeError("Invalid type of second argument.")

    result = _calculate_similarity(nodes1, nodes2, _calculate_overlap, mode)

    return [mgp.Record(node1=n1, node2=n2, similarity=similarity) for n1, n2, similarity in result]


@mgp.read_proc
def cosine(
        context: mgp.ProcCtx,
        node1: mgp.Any,
        node2: mgp.Any,
        mode: str = 'c'
) -> mgp.Record(node1=mgp.Vertex, node2=mgp.Vertex, similarity=mgp.Number):
    """
    This procedure calls the method for calculating the cosine similarity between two nodes (or two lists of nodes)
    and returns 3 fields

    :return node1: The first node
    :return node2: The second node
    :return similarity: The cosine similarity of the first and the second node

    The input arguments consist of

    :param mgp.Vertex or Tuple[mgp.Vertex] node1: A node or a tuple of nodes
    :param mgp.Vertex or Tuple[mgp.Vertex] node2: A node or a tuple od nodes
    :param str mode: Can be `p` for pairwise similarity calculation or `c` for calculating the similarity of the
    Cartesian product of given tuples. Default value is `c`.
    """

    nodes1: tuple
    nodes2: tuple

    if isinstance(node1, mgp.Vertex):
        nodes1 = tuple([node1])
    elif isinstance(node1, tuple):
        nodes1 = node1
    else:
        raise TypeError("Invalid type of first argument.")

    if isinstance(node2, mgp.Vertex):
        nodes2 = tuple([node2])
    elif isinstance(node2, tuple):
        nodes2 = node2
    else:
        raise TypeError("Invalid type of second argument.")

    result = _calculate_similarity(nodes1, nodes2, _calculate_cosine, mode)

    return [mgp.Record(node1=n1, node2=n2, similarity=similarity) for n1, n2, similarity in result]


def _calculate_jaccard(node1: mgp.Vertex, node2: mgp.Vertex) -> float:
    """
    This method calculates the Jaccard similarity between two nodes and returns their similarity.

    :return similarity: The Jaccard similarity of the first and the second node
    :rtype: float

    The input arguments consist of

    :param mgp.Vertex node1: The first node
    :param mgp.Vertex node2: The second node
    """

    jaccard_similarity = 0

    neighbours1 = _get_neighbors(node1)
    neighbours2 = _get_neighbors(node2)

    intersection_len = len(neighbours1 & neighbours2)

    denominator = (len(neighbours1) + len(neighbours2) - intersection_len)

    if denominator != 0:
        jaccard_similarity = intersection_len / denominator

    return jaccard_similarity


def _calculate_overlap(node1: mgp.Vertex, node2: mgp.Vertex) -> float:
    """
    This method calculates the overlap similarity between two nodes and returns their similarity.

    :return similarity: The overlap similarity of the first and the second node
    :rtype: float

    The input arguments consist of

    :param mgp.Vertex node1: The first node
    :param mgp.Vertex node2: The second node
    """

    overlap_similarity = 0

    neighbours1 = _get_neighbors(node1)
    neighbours2 = _get_neighbors(node2)

    denominator = (min(len(neighbours1), len(neighbours2)))

    if denominator != 0:
        overlap_similarity = len(neighbours1 & neighbours2) / denominator

    return overlap_similarity


def _calculate_cosine(node1: mgp.Vertex, node2: mgp.Vertex) -> float:
    """
    This method calculates the cosine similarity between two nodes and returns their similarity.

    :return similarity: The cosine similarity of the first and the second node
    :rtype: float

    The input arguments consist of

    :param mgp.Vertex node1: The first node
    :param mgp.Vertex node2: The second node
    """

    cosine_similarity = 0

    neighbours1 = _get_neighbors(node1)
    neighbours2 = _get_neighbors(node2)

    denominator = sqrt(len(neighbours1) * len(neighbours2))

    if denominator != 0:
        cosine_similarity = len(neighbours1 & neighbours2) / denominator

    return cosine_similarity


def _calculate_similarity(nodes1: tuple, nodes2: tuple, method: Callable, mode: str) -> List:
    """
    This method calculates the similarity of nodes with given method and mode.

    :return result: Returns the calculated similarity between nodes in a list o tuples. Each tuple consist of
    the first node, the second node and the similarity between them.
    :rtype: List[Tuple[mgp.Vertex, mgp.Vertex, float]]

    The input arguments consist of

    :param mgp.Vertex nodes1: The first tuple of nodes
    :param mgp.Vertex nodes2: The second tuple of nodes
    :param Callable method: Similarity measure which will be used for calculating the similarity between nodes.
    Currently available are `_calculate_jaccard`, `_calculate_overlap` and `_calculate_cosine`
    :param str mode: Can be `p` for pairwise similarity calculation or `c` for calculating the similarity of the
    Cartesian product of given tuples
    """

    result: list

    if not isinstance(nodes1, tuple) or not isinstance(nodes2, tuple):
        raise TypeError("Arguments should be tuples.")

    if mode not in ['p', 'c']:
        raise ValueError("Invalid mode.")

    if mode == 'p':
        if len(nodes1) == len(nodes2):
            result = [(node1, node2, method(node1, node2)) for node1, node2 in zip(nodes1, nodes2)]
        else:
            raise ValueError("Incompatible lengths of given arguments")
    elif mode == 'c':
        result = [(node1, node2, method(node1, node2)) for node1, node2 in product(nodes1, nodes2)]
    else:
        raise ValueError

    return result


def _get_neighbors(node: mgp.Vertex) -> Set[mgp.Vertex]:
    """
    This method find all neighbors of a given node.

        :return neighbors: All neighbors of a node
        :rtype: Set

    The input arguments consist of

        :param mgp.Vertex node: A node
    """

    neighbors = set()

    if isinstance(node, mgp.Vertex):
        for edge in node.in_edges:
            neighbors.add(edge.from_vertex)

        for edge in node.out_edges:
            neighbors.add(edge.to_vertex)
    else:
        raise TypeError("Argument type must be mgp.Vertex.")

    return neighbors
