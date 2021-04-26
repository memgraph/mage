import pytest

from mage.graph_coloring_module import Individual
from mage.graph_coloring_module import ChainPopulation
from mage.graph_coloring_module import Graph


@pytest.fixture
def chain_population():
    graph = Graph(
        [0, 1, 2, 3, 4],
        {
            0: [(1, 2), (2, 3)],
            1: [(0, 2), (2, 2), (4, 5)],
            2: [(0, 3), (1, 2), (3, 3)],
            3: [(2, 3)],
            4: [(1, 5)],
        },
    )
    indv_1 = Individual(
        num_of_colors=3, graph=graph, chromosome=[1, 1, 0, 2, 0], conflict_nodes={0, 1}
    )
    indv_2 = Individual(num_of_colors=3, graph=graph, chromosome=[1, 2, 0, 2, 1])
    indv_3 = Individual(
        num_of_colors=3, graph=graph, chromosome=[2, 1, 0, 2, 1], conflict_nodes={1, 4}
    )
    population = ChainPopulation(graph, [indv_1, indv_2, indv_3])
    return population


def test_previous_individual(chain_population):
    result_indv = chain_population.get_prev_individual(2)
    expected_indv = chain_population[1]

    assert result_indv == expected_indv


def test_previous_negative_index(chain_population):
    with pytest.raises(IndexError):
        chain_population.get_prev_individual(-2)


def test_previous_out_of_range(chain_population):
    with pytest.raises(IndexError):
        chain_population.get_prev_individual(10)


def test_previous_first_item(chain_population):
    result_indv = chain_population.get_prev_individual(0)
    expected_indv = chain_population[2]

    assert result_indv == expected_indv


def test_next_individual(chain_population):
    result_indv = chain_population.get_next_individual(0)
    expected_indv = chain_population[1]

    assert result_indv == expected_indv


def test_next_individual_last_item(chain_population):
    result_indv = chain_population.get_next_individual(2)
    expected_indv = chain_population[0]

    assert result_indv == expected_indv


def test_next_negative_index(chain_population):
    with pytest.raises(IndexError):
        chain_population.get_next_individual(-2)


def test_next_out_of_range(chain_population):
    result_indv = chain_population.get_next_individual(2)
    expected_indv = chain_population[0]

    assert result_indv == expected_indv
