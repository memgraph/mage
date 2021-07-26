from typing import List, Dict

INITIAL_RANK = 0


class Node:
    """
    Class implementing a node in an union-find data structure.
    Stores the current node's rank and a reference to its parent node.
    """

    def __init__(self, parent_id: int, rank: int = INITIAL_RANK):
        self._parent = parent_id
        self._rank = rank

    @property
    def parent(self) -> int:
        return self._parent

    @parent.setter
    def parent(self, x: int):
        self._parent = x

    @property
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, x: int):
        self._rank = x


class UnionFind:
    """
    Class implementing a disjoint-set data structure.
    """

    def __init__(self, node_ids: List[int]):
        self.nodes: Dict[int, Node] = {node_id: Node(parent_id=node_id, rank=INITIAL_RANK) for node_id in node_ids}

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

    def union(self, node1_ids: List[int], node2_ids: List[int]) -> None:
        """
        Executes a sequence of union operations on each pair of same-index nodes from two equal-length lists.

        :param node1_ids: First list of node IDs
        :param node2_ids: Second list of node IDs
        """
        for node1_id, node2_id in zip(node1_ids, node2_ids):
            self.union_pair(node1_id, node2_id)

    def union_pair(self, node1_id, node2_id) -> None:
        """
        Unites the components containing two given nodes. Implements union by rank to reduce component tree height.

        :param node1_id: First node's ID
        :param node2_id: Second node's ID
        """
        root1 = self.find(node1_id)
        root2 = self.find(node2_id)

        if root1 == root2:
            return

        if self.rank(root1) < self.rank(root2):
            root1, root2 = root2, root1

        self.nodes[root2].parent = root1
        if self.rank(root1) == self.rank(root2):
            self.nodes[root1].rank = self.rank(root1) + 1

    def connected(self, node1_ids: List[int], node2_ids: List[int]) -> List[bool]:
        """
        Returns whether nodes belong to the same connected component for each pair of same-index nodes from two
        equal-length lists.

        :param node1_ids: First list of node IDs
        :param node2_ids: Second list of node IDs
        """
        return [self.connected_pair(node1_id, node2_id) for node1_id, node2_id in zip(node1_ids, node2_ids)]

    def connected_pair(self, node1_id: int, node2_id: int) -> bool:
        """
        Returns whether two given nodes belong to the same connected component.

        :param node1_id: First node's ID
        :param node2_id: Second node's ID
        """
        return self.find(node1_id) == self.find(node2_id)
