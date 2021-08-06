from enum import Enum
from math import sqrt
from typing import Set, List, Callable, Tuple, Union
from itertools import product
import mgp

# not implemented for directed graphs
# not implemented for multigraphs
# not taking into account edge weights

neighbors_dict = dict()


class Mode(Enum):
    """
    Enum for mode parameter
    """

    CARTESIAN = "cartesian"
    PAIRWISE = "pairwise"


@mgp.read_proc
def jaccard(
    context: mgp.ProcCtx, node1: mgp.Any, node2: mgp.Any, mode: str = "cartesian"
) -> mgp.Record(node1=mgp.Vertex, node2=mgp.Vertex, similarity=float):
    """This procedure calls a method for calculating the similarity between two nodes (or two tuples of nodes)
    so that the used node similarity measure is the Jaccard similarity.

    :param node1: A node or a tuple of nodes
    :type node1: mgp.Vertex, tuple
    :param node2: A node or a tuple of nodes
    :type node2: mgp.Vertex, tuple
    :param mode: Can be `pairwise` for pairwise similarity calculation or `cartesian`
    for calculating the similarity of the Cartesian product of given tuples.
    The default value is set to be a Cartesian product.
    :type mode: str, optional

    :return: This procedure returns a mgp.Record with 3 fields:
    the first node, the second node and the Jaccard similarity between them
    :rtype: mgp.Record
    """

    return _calculate_similarity(node1, node2, _calculate_jaccard, Mode(mode))


@mgp.read_proc
def overlap(
    context: mgp.ProcCtx, node1: mgp.Any, node2: mgp.Any, mode: str = "cartesian"
) -> mgp.Record(node1=mgp.Vertex, node2=mgp.Vertex, similarity=mgp.Number):
    """This procedure calls a method for calculating the similarity between two nodes (or two tuples of nodes)
    so that the used node similarity measure is the overlap similarity.

    :param node1: A node or a tuple of nodes
    :type node1: mgp.Vertex, tuple
    :param node2: A node or a tuple of nodes
    :type node2: mgp.Vertex, tuple
    :param mode: Can be `pairwise` for pairwise similarity calculation or `cartesian`
    for calculating the similarity of the Cartesian product of given tuples.
    The default value is set to be a Cartesian product.
    :type mode: str, optional

    :return: This procedure returns a mgp.Record with 3 fields:
    the first node, the second node and the overlap similarity between them
    :rtype: mgp.Record
    """

    return _calculate_similarity(node1, node2, _calculate_overlap, Mode(mode))


@mgp.read_proc
def cosine(
    context: mgp.ProcCtx, node1: mgp.Any, node2: mgp.Any, mode: str = "cartesian"
) -> mgp.Record(node1=mgp.Vertex, node2=mgp.Vertex, similarity=mgp.Number):
    """This procedure calls a method for calculating the similarity between two nodes (or two tuples of nodes)
    so that the used node similarity measure is the cosine similarity.

    :param node1: A node or a tuple of nodes
    :type node1: mgp.Vertex, tuple
    :param node2: A node or a tuple of nodes
    :type node2: mgp.Vertex, tuple
    :param mode: Can be `pairwise` for pairwise similarity calculation or `cartesian`
    for calculating the similarity of the Cartesian product of given tuples.
    The default value is set to be a Cartesian product.
    :type mode: str, optional

    :return: This procedure returns a mgp.Record with 3 fields:
    the first node, the second node and the cosine similarity between them
    :rtype: mgp.Record
    """

    return _calculate_similarity(node1, node2, _calculate_cosine, Mode(mode))


def _calculate_jaccard(node1: mgp.Vertex, node2: mgp.Vertex) -> float:
    """This method calculates the Jaccard similarity between two nodes.

    :param node1: The first node
    :type node1: mgp.Vertex
    :param node2: The second node
    :type node2: mgp.Vertex

    :return similarity: The Jaccard similarity of the first and the second node
    :rtype: float
    """

    jaccard_similarity: float

    neighbors1: set
    neighbors2: set

    if node1 == node2:
        return 1.0

    jaccard_similarity = 0.0

    neighbors1 = _get_neighbors(node1)
    neighbors2 = _get_neighbors(node2)

    intersection_len = len(neighbors1 & neighbors2)
    denominator = len(neighbors1) + len(neighbors2) - intersection_len

    if denominator != 0:
        jaccard_similarity = intersection_len / denominator

    return jaccard_similarity


def _calculate_overlap(node1: mgp.Vertex, node2: mgp.Vertex) -> float:
    """This method calculates the overlap similarity between two nodes.

    :param node1: The first node
    :type node1: mgp.Vertex
    :param node2: The second node
    :type node2: mgp.Vertex

    :return: The overlap similarity of the first and the second node
    :rtype: float

    """

    overlap_similarity: float

    neighbors1: set
    neighbors2: set

    if node1 == node2:
        return 1.0

    overlap_similarity = 0.0

    neighbors1 = _get_neighbors(node1)
    neighbors2 = _get_neighbors(node2)

    denominator = min(len(neighbors1), len(neighbors2))

    if denominator != 0:
        overlap_similarity = len(neighbors1 & neighbors2) / denominator

    return overlap_similarity


def _calculate_cosine(node1: mgp.Vertex, node2: mgp.Vertex) -> float:
    """This method calculates the cosine similarity between two nodes.

    :param node1: The first node
    :type node1: mgp.Vertex
    :param node2: The second node
    :type node2: mgp.Vertex

    :return: The cosine similarity of the first and the second node
    :rtype: float
    """

    cosine_similarity: float

    neighbors1: set
    neighbors2: set

    if node1 == node2:
        return 1.0

    cosine_similarity = 0.0

    neighbors1 = _get_neighbors(node1)
    neighbors2 = _get_neighbors(node2)

    denominator = sqrt(len(neighbors1) * len(neighbors2))

    if denominator != 0:
        cosine_similarity = len(neighbors1 & neighbors2) / denominator

    return cosine_similarity


def _calculate_similarity(
    node1: Union[mgp.Vertex, Tuple[mgp.Vertex]],
    node2: Union[mgp.Vertex, Tuple[mgp.Vertex]],
    method: Callable,
    mode: Mode,
) -> List[Tuple[mgp.Vertex, mgp.Vertex, float]]:
    """This method calculates the similarity of nodes for given method and mode.

    :param node1: The first node or tuple of nodes
    :type node1: mgp.Vertex, tuple
    :param node2: The second node or tuple of nodes
    :type node2: mgp.Vertex, tuple
    :param method: Similarity measure which will be used for calculating the similarity between nodes.
    Currently available are `_calculate_jaccard`, `_calculate_overlap` and `_calculate_cosine`
    :type method: function
    :param mode: Can be PAIRWISE for pairwise similarity calculation or CARTESIAN
    for calculating the similarity of the Cartesian product of given tuples.
    :type mode: enum

    :raises TypeError: Occurs if there's a type mismatch for passed arguments
    :raises ValueError: Occurs if any passed argument is invalid

    :return: Returns the calculated similarity between nodes in a list of mgp.Records. Each mgp.Record consist of
    the first node, the second node and the similarity between them.
    :rtype: list
    """

    # clear cache
    neighbors_dict.clear()

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

    if mode == Mode.PAIRWISE:
        if len(nodes1) == len(nodes2):
            return [
                mgp.Record(node1=n1, node2=n2, similarity=method(n1, n2))
                for n1, n2 in zip(nodes1, nodes2)
            ]
        else:
            raise ValueError("Incompatible lengths of given arguments.")
    elif mode == Mode.CARTESIAN:
        return [
            mgp.Record(node1=n1, node2=n2, similarity=method(n1, n2))
            for n1, n2 in product(nodes1, nodes2)
        ]
    else:
        raise ValueError("Invalid mode.")


def _get_neighbors(node: mgp.Vertex) -> Set[int]:
    """This method finds all neighbors of a given node. If neighbors of a node have already been fetched once before,
    they can be found in the neighbors_dict dictionary and thus the method returns the neighbors faster.
    neighbors_dict is a global variable and it resets at the beginning of the program

    :param node: A node
    :type node: mgp.Vertex

    :return: Set of all IDs of neighbors of a node
    :rtype: set
    """

    if node.id in neighbors_dict:
        return neighbors_dict[node.id]

    neighbors = set()

    for edge in node.in_edges:
        neighbors.add(edge.from_vertex.id)

    for edge in node.out_edges:
        neighbors.add(edge.to_vertex.id)

    neighbors_dict[node.id] = neighbors

    return neighbors
