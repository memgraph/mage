import time

import pytest

from mage.node2vec_online_module.walk_sampling import StreamWalkUpdater

EMBEDDINGS_DIM = 2
INCORRECT_NEGATIVE_RATE = -1


@pytest.fixture
def walk_sampling():
    return StreamWalkUpdater(
        half_life=7200,
        max_length=3,
        beta=0.9,
        cutoff=604800,
        sampled_walks=2,
        full_walks=False,
    )


def test_correct_walk_number(walk_sampling):
    walks = walk_sampling.process_new_edge(0, 1, time.time())
    assert len(walks) == 2


def test_legal_combinations_time_crossover_edges(walk_sampling):
    walk_sampling.process_new_edge(1, 2, time.time())
    walk_sampling.process_new_edge(0, 1, time.time())

    walks = walk_sampling.process_new_edge(2, 3, time.time())

    # in time first goes 1->2 then 0->1 and then 2->3, legal combinations are 1,3 and 2,3 since 0->1 appear after 1->2

    for combination in walks:
        assert combination[0] == 2 or combination[0] == 1
        assert combination[1] == 3


def test_legal_combinations_time_linear_edges(walk_sampling):
    walk_sampling.process_new_edge(0, 1, time.time())
    walk_sampling.process_new_edge(1, 2, time.time())
    walks = walk_sampling.process_new_edge(2, 3, time.time())

    # legal combinations are 0,3, 1,3 and 2,3 since 0->1 appear after 1->2

    for combination in walks:
        assert combination[0] == 2 or combination[0] == 1 or combination[0] == 0
        assert combination[1] == 3
