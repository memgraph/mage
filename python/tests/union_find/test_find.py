from python.union_find import UnionFind
import pytest

IDs = [i for i in range(10)]


@pytest.fixture
def ds():
    return UnionFind(node_ids=IDs)


class TestFind:
    def test_disconnected(self, ds):
        assert ds.connected([0], [1]) == [False]

    def test_connected(self, ds):
        ds.union([0], [1])
        assert ds.connected([0], [1]) == [True]

    def test_connected_later(self, ds):
        ds.union([0, 1], [1, 2])
        assert ds.connected([0], [2]) == [True]
