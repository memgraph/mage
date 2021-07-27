from python.mage.union_find.union_find import UnionFind
import pytest

IDs = [i for i in range(10)]


@pytest.fixture
def ds():
    return UnionFind(node_ids=IDs)


class TestFind:
    def test_disconnected(self, ds):
        assert ds.connected(0, 1) is False

    def test_connected(self, ds):
        ds.union(0, 1)
        assert ds.connected(0, 1) is True

    def test_connected_later(self, ds):
        ds.union([0, 1], [1, 2])
        assert ds.connected(0, 2) is True
