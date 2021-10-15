import os

import pytest

from mage.node2vec.graph import BasicGraph

WORK_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATASET = f"{WORK_DIRECTORY}/edges.txt"

EDGES_WEIGHTS = {
    (0, 1): 0.5,
    (0, 2): 1,
    (0, 4): 1,
    (1, 5): 1,
    (1, 6): 1,
    (2, 5): 1,
    (3, 0): 1,
    (4, 5): 1,
    (7, 1): 1,
    (7, 5): 1,
    (7, 6): 1,
}


@pytest.fixture(params=[True, False])
def is_directed(request):
    return request.param


@pytest.fixture
def edges_dataset():
    print(DATASET)
    with open(DATASET) as fp:
        edges_dataset = fp.readlines()
    return edges_dataset


@pytest.fixture
def basic_graph(edges_dataset, is_directed) -> BasicGraph:
    edges_weights = {}
    for line in edges_dataset:
        line_parts = line.split(" ")
        edges_weights[(int(line_parts[0]), int(line_parts[1]))] = 1

    return BasicGraph(edges_weights, is_directed)


@pytest.fixture
def basic_graph_from_dict(is_directed) -> BasicGraph:
    return BasicGraph(EDGES_WEIGHTS, is_directed)


def test_graph_edges(edges_dataset, basic_graph):
    graph_edges = basic_graph.get_edges()
    if basic_graph.is_directed:
        assert len(graph_edges) == len(edges_dataset)
    else:
        assert len(graph_edges) == len(edges_dataset) * 2

    for edge in edges_dataset:
        edge_nodes = edge.split(" ")
        if basic_graph.is_directed:
            assert basic_graph.has_edge(int(edge_nodes[0]), int(edge_nodes[1]))
            assert not basic_graph.has_edge(int(edge_nodes[1]), int(edge_nodes[0]))
        if not basic_graph.is_directed:
            assert basic_graph.has_edge(int(edge_nodes[0]), int(edge_nodes[1]))
            assert basic_graph.has_edge(int(edge_nodes[1]), int(edge_nodes[0]))


def test_graph_edges_from_dict(basic_graph_from_dict):
    graph_edges = basic_graph_from_dict.get_edges()
    if basic_graph_from_dict.is_directed:
        assert len(graph_edges) == len(EDGES_WEIGHTS)
    else:
        assert len(graph_edges) == len(EDGES_WEIGHTS) * 2

    if basic_graph_from_dict.is_directed:
        assert basic_graph_from_dict.get_neighbors(0) == [1, 2, 4]
    else:
        assert basic_graph_from_dict.get_neighbors(0) == [1, 2, 3, 4]

    if not basic_graph_from_dict.is_directed:
        assert basic_graph_from_dict.get_edge_weight(0, 1) == 0.5
        assert basic_graph_from_dict.get_edge_weight(1, 0) == 0.5

    if basic_graph_from_dict.is_directed:
        assert basic_graph_from_dict.has_edge(0, 3) is False
        assert basic_graph_from_dict.has_edge(3, 0) is True
    else:
        assert basic_graph_from_dict.has_edge(0, 3) is True
        assert basic_graph_from_dict.has_edge(3, 0) is True
