from typing import Dict, List, Union
from mage.union_find.node import Node, INITIAL_RANK


class UnionFind:
    """
    Class implementing a disjoint-set data structure.
    """

    def __init__(self, node_ids: List[int]):
        self.nodes: Dict[int, Node] = {
            node_id: Node(parent_id=node_id, rank=INITIAL_RANK) for node_id in node_ids
        }

    def parent(self, node_id: int) -> int:
        """
        Returns given node's parent's ID.

        :param node_id: Node ID
        """
        return self.nodes[node_id].parent

    def grandparent(self, node_id: int) -> int:
        """
        Returns given node's grandparent's ID.

        :param node_id: Node ID
        """
        return self.parent(self.parent(node_id))

    def rank(self, node_id: int) -> int:
        """
        Returns given node's rank.

        :param node_id: Node ID
        """
        return self.nodes[node_id].rank

    def find(self, node_id: int) -> int:
        """
        Returns the representative node's ID for the component that given node is member of.
        Uses path splitting (https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Finding_set_representatives)
        in order to keep trees representing connected components flat.

        :param node_id: Node ID
        """
        while node_id != self.parent(node_id):
            self.nodes[node_id].parent = self.grandparent(node_id)  # path splitting
            node_id = self.parent(node_id)

        return node_id

    def union(
        self, node_id_1: Union[int, List[int]], node_id_2: Union[int, List[int]]
    ) -> None:
        """
        Executes a sequence of union operations on each pair of same-index nodes.
        Supports two nodes or two equal-length lists thereof.

        :param node_id_1: First node's ID or list of node IDs
        :param node_id_2: Second node's ID or list of node IDs
        """
        if isinstance(node_id_1, int) and isinstance(node_id_2, int):
            self.union_pair(node_id_1, node_id_2)
        else:
            for x, y in zip(node_id_1, node_id_2):
                self.union_pair(node_id_1=x, node_id_2=y)

    def union_pair(self, node_id_1: int, node_id_2: int) -> None:
        """
        Unites the components containing two given nodes. Implements union by rank to reduce component tree height.

        :param node_id_1: First node's ID
        :param node_id_2: Second node's ID
        """
        root_1 = self.find(node_id_1)
        root_2 = self.find(node_id_2)

        if root_1 == root_2:
            return

        if self.rank(root_1) < self.rank(root_2):
            root_1, root_2 = root_2, root_1

        self.nodes[root_2].parent = root_1
        if self.rank(root_1) == self.rank(root_2):
            self.nodes[root_1].rank = self.rank(root_1) + 1

    def connected(
        self, node_id_1: Union[int, List[int]], node_id_2: Union[int, List[int]]
    ) -> Union[bool, List[bool]]:
        """
        Returns whether nodes belong to the same connected component for each pair of same-index nodes.
        Supports two nodes or two equal-length lists thereof.

        :param node_id_1: First node's ID or list of node IDs
        :param node_id_2: Second node's ID or list of node IDs
        """
        if isinstance(node_id_1, int) and isinstance(node_id_2, int):
            return self.connected_pair(node_id_1, node_id_2)
        else:
            return [
                self.connected_pair(node_id_1=x, node_id_2=y)
                for x, y in zip(node_id_1, node_id_2)
            ]

    def connected_pair(self, node_id_1: int, node_id_2: int) -> bool:
        """
        Returns whether two given nodes belong to the same connected component.

        :param node_id_1: First node's ID
        :param node_id_2: Second node's ID
        """
        return self.find(node_id_1) == self.find(node_id_2)
