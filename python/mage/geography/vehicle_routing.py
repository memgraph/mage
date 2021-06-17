from abc import ABC, abstractmethod
import numpy as np
from typing import Iterator, List


class VRPResult:
    """
    The VRP Result consists of multiple VRP paths.
    """

    def __init__(self, routes: List["VRPRoute"]):
        self.routes = routes


class VRPRoute:
    def __init__(self, paths: List[int]):
        self.paths = paths

        # Smallest path scenario is DEPOT - NODE - DEPOT
        if len(self.paths) < 3:
            raise InvalidRouteException(
                f"Length of route is {len(self.paths)}, but smallest amount is 3!"
            )

        if self.paths[0] != self.paths[-1]:
            raise InvalidRouteException(
                "Cannot have different starting and ending points in route!"
            )

    def __getitem__(self, index) -> int:
        if index < 0 or index >= len(self.paths):
            raise IndexError
        return self.paths[index]

    def __iter__(self) -> Iterator[int]:
        yield from self.paths

    def __contains__(self, item: int) -> bool:
        return item in self.paths

    def __len__(self) -> int:
        return len(self.paths)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, VRPRoute):
            return self.paths == o.paths
        return False

    def __repr__(self) -> str:
        return str(self.paths)


class VRPSolver(ABC):
    """
    VRP Solver solves the VRP problem and can extract results to desired hook.
    """

    @abstractmethod
    def solve(self):
        """
        Implementation method.
        """
        pass

    @abstractmethod
    def get_result(self):
        """
        Extract results from solved problem.
        """
        pass


def get_route_length(vrp_route: VRPRoute, distance_matrix: np.ndarray):
    """
    Returns the length of a route using the distance matrix.
    """
    length = 0.0
    for i in range(len(vrp_route.paths) - 1):
        (from_vertex, to_vertex) = vrp_route[i], vrp_route[i + 1]
        length += distance_matrix[from_vertex][to_vertex]

    return length


class InvalidDepotException(Exception):
    pass


class InvalidRouteException(Exception):
    pass
