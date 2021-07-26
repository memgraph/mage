from python.union_find import UnionFind
import pytest

IDs = [i for i in range(10)]


@pytest.fixture
def ds():
    return UnionFind(node_ids=IDs)


class TestInit:
    def test_keys(self, ds):
        assert all(i in ds.nodes.keys() for i in IDs)

    def test_parent(self, ds):
        assert all(ID == ds.nodes[ID].parent for ID in IDs)

    def test_rank(self, ds):
        assert all(ds.nodes[ID].rank == 0 for ID in IDs)
