from mage.union_find.union_find import UnionFind
import pytest

IDs = [i for i in range(10)]


@pytest.fixture
def ds():
    return UnionFind(node_ids=IDs)


class TestUnion:
    def test_equal_height(self, ds):
        ds.union(0, 1)
        assert ds.nodes[1].parent == 0

    def test_different_height(self, ds):
        ds.union([0, 1], [1, 2])
        assert ds.nodes[2].parent == 0
