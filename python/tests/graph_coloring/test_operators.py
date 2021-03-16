import random
import pytest

from mage.graph_coloring_module import Graph
from mage.graph_coloring_module import Individual
from mage.graph_coloring_module import MISMutation
from mage.graph_coloring_module import MultipleMutation
from mage.graph_coloring_module import RandomMutation
from mage.graph_coloring_module import SimpleMutation

@pytest.fixture
def set_seed():
    random.seed(42)

@pytest.fixture
def graph():
    return  Graph (
        [0, 1, 2, 3, 4],
        {
            0 : [(1, 2), (2, 3)],
            1 : [(0, 2), (2, 2), (4, 5)],
            2 : [(0, 3), (1, 2), (3, 3)],
            3 : [(2, 3)],
            4 : [(1, 5)]
        })


def test_mis_mutation(set_seed, graph):
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[0, 1, 0, 2, 0])
    mutated_indv, nodes = MISMutation().mutate(graph, individual)

    expected_mutated_indv_chromosome = [0, 2, 0, 2, 0]
    expected_nodes = [3, 1]
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)


def test_multiple_mutation(set_seed, graph):
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[0, 1, 0, 2, 0])
    mutated_indv, nodes = MultipleMutation().mutate(graph, individual, {
        "multiple_mutation_no_of_nodes": 2,
        "no_of_colors": 3
    })

    expected_mutated_indv_chromosome = [1, 1, 0, 2, 0]
    expected_nodes = [0, 0]
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)


def test_random_mutation_no_conflict(set_seed, graph):
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[0, 1, 2, 1, 0])
    mutated_indv, nodes = RandomMutation().mutate(graph, individual, {
        "random_mutation_probability": 0.5,
        "random_mutation_probability_2": 0.5,
        "no_of_colors": 3
    })
    expected_mutated_indv_chromosome = [0, 1, 2, 1, 0]
    expected_nodes = []
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)


def test_random_mutation(set_seed, graph):
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[1, 1, 0, 2, 0], conflict_nodes={0, 1})
    mutated_indv, nodes = RandomMutation().mutate(graph, individual, {
        "random_mutation_probability": 0.5,
        "random_mutation_probability_2": 0.5,
        "no_of_colors": 3
    })
    expected_mutated_indv_chromosome = [2, 1, 0, 2, 0]
    expected_nodes = [0]
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)


def test_simple_no_conflict(set_seed, graph):
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[0, 1, 2, 1, 0])
    mutated_indv, nodes = SimpleMutation().mutate(graph, individual)
    expected_mutated_indv_chromosome = [0, 1, 2, 1, 0]
    expected_nodes = []
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)


def test_simple_mutation(set_seed, graph):
    individual = Individual(no_of_colors=3, graph=graph, chromosome=[1, 1, 0, 2, 0], conflict_nodes={0, 1})
    mutated_indv, nodes = SimpleMutation().mutate(graph, individual)
    expected_mutated_indv_chromosome = [0, 1, 0, 2, 0]
    expected_nodes = [0]
    assert mutated_indv.chromosome == expected_mutated_indv_chromosome
    assert sorted(nodes) == sorted(expected_nodes)