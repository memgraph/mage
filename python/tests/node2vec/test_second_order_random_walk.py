import numpy as np
import pytest

from mage.node2vec.graph import BasicGraph
from mage.node2vec.second_order_random_walk import SecondOrderRandomWalk

EDGES_DICT_SAME_WEIGHTS = {
    (0, 1): 1,
    (0, 2): 1,
    (0, 4): 1,
    (1, 5): 1,
    (1, 6): 1,
    (2, 5): 1,
    (3, 0): 1,
    (4, 6): 1,
    (7, 1): 1,
    (7, 5): 1,
    (7, 6): 1,
    # new edges
    (0, 5): 1,
    (6, 0): 1,
}

UNDIRECT_GRAPH_EDGE_TRANSITION_PROBS = {
    (0, 1): [0.11111, 0.22222, 0.22222, 0.44444],
    (3, 0): [0.19047619, 0.19047619, 0.04761905, 0.19047619, 0.19047619, 0.19047619],
}

DIRECT_GRAPH_EDGE_TRANSITION_PROBS = {
    (0, 1): [0.66667, 0.33333],
    (0, 3): [0.33333, 0.33333],
}


def get_basic_graph(dataset, is_directed) -> BasicGraph:
    return BasicGraph(dataset, is_directed)


@pytest.mark.parametrize(
    "dataset, is_directed",
    [
        (EDGES_DICT_SAME_WEIGHTS, True),
        (EDGES_DICT_SAME_WEIGHTS, False),
    ],
)
def test_graph_transition_probs(dataset, is_directed):
    basic_graph = get_basic_graph(dataset, is_directed)
    if basic_graph.is_directed:
        graph_transition_probs = DIRECT_GRAPH_EDGE_TRANSITION_PROBS
    else:
        graph_transition_probs = UNDIRECT_GRAPH_EDGE_TRANSITION_PROBS

    second_order_random_walk = SecondOrderRandomWalk(
        p=2, q=0.5, walk_length=3, num_walks=2
    )
    second_order_random_walk.set_graph_transition_probs(basic_graph)
    for edge in basic_graph.get_edges():
        if edge not in graph_transition_probs:
            continue

        calculated_transition_probs = basic_graph.get_edge_transition_probs(edge)

        correct_transition_probs = graph_transition_probs.get(edge)

        assert np.all(
            np.isclose(
                calculated_transition_probs,
                correct_transition_probs,
                atol=1e-5,
            )
        )
