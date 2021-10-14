from abc import ABC, abstractmethod
import numpy as np
from typing import Iterator, List


class VRPResult:
    """
    The VRP Result consists of multiple VRP paths.
    """

    def __init__(self, routes: List["VRPRoute"]):
        self.routes = routes

    def __repr__(self) -> str:
        return str(self.routes)


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


def get_vrp_result_length(vrp_result: VRPResult, distance_matrix: np.ndarray):
    """
    Returns the length of the whole VRP Result using the distance matrix
    """
    length = 0.0
    for route in vrp_result.routes:
        length += get_route_length(route, distance_matrix)

    return length


def get_route_length(vrp_route: VRPRoute, distance_matrix: np.ndarray):
    """
    Returns the length of a route using the distance matrix.
    """
    length = 0.0
    for i in range(len(vrp_route.paths) - 1):
        (from_vertex, to_vertex) = vrp_route[i], vrp_route[i + 1]
        length += distance_matrix[from_vertex][to_vertex]

    return length


def is_location_contained_in_result(vrp_result: VRPResult, location_id: int) -> bool:
    return location_id in [x for routes in vrp_result.routes for x in routes.paths]


class VRPDynamicSolutionIterator:
    def __init__(self, paths: List[int], new_location: int) -> None:
        self.paths = paths
        self.new_location = new_location
        self.current_path_index = 0
        self.current_position_index = 1

    def __iter__(self):
        return self

    def __next__(self) -> VRPResult:
        if self.current_path_index >= len(self.paths):
            raise StopIteration

        result = self._next_element()

        self._update_indexes()

        return result

    def _update_indexes(self):
        path_length = len(self.paths[self.current_path_index])

        if self.current_position_index == path_length - 1:
            self.current_path_index += 1
            self.current_position_index = 1
        else:
            self.current_position_index += 1

    def _next_element(self) -> VRPResult:
        dynamic_paths = []
        for (index, path) in enumerate(self.paths):
            if index != self.current_path_index:
                dynamic_paths.append(path)
                continue

            modified_path = (
                path[: self.current_position_index]
                + [self.new_location]
                + path[self.current_position_index :]
            )
            dynamic_paths.append(modified_path)

        return VRPResult([VRPRoute(x) for x in dynamic_paths])


class DynamicRouting(VRPSolver):
    def __init__(
        self,
        distance_matrix: np.ndarray,
        initial_vrp_result: VRPResult,
        new_locations: List[int],
    ) -> None:
        self.distance_matrix = distance_matrix
        self.initial_vrp_result = initial_vrp_result
        self.new_locations = new_locations

    def solve(self):
        result = self.initial_vrp_result
        for new_location in self.new_locations:
            result = self.reroute(result, new_location)

        self.vrp_result = result

    def get_result(self):
        if self.vrp_result is None:
            raise VRPResultNotPresent("VRP Result not yet present!")

        return self.vrp_result

    def reroute(self, vrp_routes: VRPResult, new_location_idx: int) -> VRPResult:
        paths = []
        [paths.append(route.paths.copy()) for route in vrp_routes.routes]

        dynamic_solution_iterator = VRPDynamicSolutionIterator(paths, new_location_idx)

        best_solution = None
        best_distance = None
        for modified_vrp_result in dynamic_solution_iterator:
            distance = get_vrp_result_length(modified_vrp_result, self.distance_matrix)
            if best_solution is None or distance < best_distance:
                best_solution = modified_vrp_result
                best_distance = distance

        return best_solution


class InvalidDepotException(Exception):
    pass


class InvalidRouteException(Exception):
    pass


class VRPResultNotPresent(Exception):
    pass
