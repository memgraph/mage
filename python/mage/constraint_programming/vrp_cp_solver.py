from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from mage.geography.vehicle_routing import VRPRoute, get_route_length
from gekko import GEKKO
from mage.geography import VRPResult, VRPSolver, InvalidDepotException
from typing import Dict, List, Tuple
import numpy as np


"""
VRP Path is an edge from a starting to ending node
"""
VRPPath = namedtuple("VRPPath", ("from_vertex, to_vertex"))


class VRPConstraintProgrammingSolver(VRPSolver):
    """
    This constraint solver solves the Vehicle Routing Problem with constraint programming using GEKKO.
    """

    SOURCE_INDEX = -1
    SINK_INDEX = -2

    def __init__(self, no_vehicles: int, distance_matrix: np.array, depot_index: int):
        if depot_index < 0 or depot_index >= len(distance_matrix):
            raise InvalidDepotException("Depot index outside the range of locations!")

        self._model = GEKKO(remote=False)

        self.no_vehicles = no_vehicles
        self.distance_matrix = distance_matrix
        self.depot_index = depot_index

        self._edge_chosen_vars = dict()
        self._time_vars = dict()
        self._location_node_ids = [
            x for x in range(len(distance_matrix)) if x != self.depot_index
        ]

        self._constraints: List[VRPConstraint] = [
            TimeIncreasesWithPassingFromOneNodeToAnotherConstraint(
                self._model,
                self._edge_chosen_vars,
                self._time_vars,
                self.distance_matrix,
            ),
            No3NodeCyclesConstraint(
                self._model,
                self._edge_chosen_vars,
                self._location_node_ids,
            ),
            StartInSourceNodeConstraint(
                self._model,
                self._edge_chosen_vars,
                self._location_node_ids,
                self.no_vehicles,
                self.SOURCE_INDEX,
            ),
            EndInSinkNodeConstraint(
                self._model,
                self._edge_chosen_vars,
                self._location_node_ids,
                self.no_vehicles,
                self.SINK_INDEX,
            ),
            MaximumEdgesActivatedConstraint(
                self._model,
                self._edge_chosen_vars,
                self._location_node_ids,
                self.no_vehicles,
            ),
            NoBacktrackingConstraint(self._model, self._edge_chosen_vars),
        ]

        self._initialize()
        self._add_constraints()
        self._add_objective()
        self._add_options()

    def solve(self):
        self._model.solve()

    def get_result(self) -> VRPResult:
        individual_vrp_paths = [
            VRPPath(
                key[0] if key[0] >= 0 else self.depot_index,
                key[1] if key[1] >= 0 else self.depot_index,
            )
            for key, var in self._edge_chosen_vars.items()
            if int(var.value[0]) == 1
        ]

        routes = VRPCPRouteValidation(
            individual_vrp_paths, self.distance_matrix, self.depot_index
        ).create_routes()

        return VRPResult(routes)

    def get_distance(self, edge: Tuple[int, int]) -> float:
        node_from, node_to = edge

        if any(
            node in [self.SOURCE_INDEX, self.SINK_INDEX]
            for node in [node_from, node_to]
        ):
            return 0

        return self.distance_matrix[node_from][node_to]

    def _initialize(self):
        for node_index in range(len(self.distance_matrix)):
            if node_index in self._location_node_ids:
                self._initialize_location_node(node_index)

    def _initialize_location_node(self, node_index: int):
        self._time_vars[node_index] = self._model.Var(value=0, lb=0, integer=False)

        # Initialize starting point and sinking point for every vehicle
        self._add_variable((self.SOURCE_INDEX, node_index))
        self._add_variable((node_index, self.SINK_INDEX))

        # For every node, draw lengths from and to it, with duration of edges
        out_vars = self._add_adjacent_output_edge_variables(node_index)
        in_vars = self._add_adjacent_input_edge_variables(node_index)

        # Either it was a beginning node, or a vehicle has visited it in the drive.
        if len(out_vars) > 0:
            self._model.Equation(
                self._edge_chosen_vars[(node_index, self.SINK_INDEX)] + sum(out_vars)
                == 1
            )

        if len(in_vars) > 0:
            self._model.Equation(
                self._edge_chosen_vars[(self.SOURCE_INDEX, node_index)] + sum(in_vars)
                == 1
            )

    def _add_adjacent_output_edge_variables(
        self, node_index: int
    ) -> List[Tuple[int, int]]:
        edges_vars = []

        for adjacent_node in range(len(self.distance_matrix)):
            if adjacent_node == self.depot_index:
                continue

            edge = (node_index, adjacent_node)
            var = self._add_variable(edge)
            edges_vars.append(var)

        return edges_vars

    def _add_adjacent_input_edge_variables(
        self, node_index: int
    ) -> List[Tuple[int, int]]:
        edges_vars = []

        for adjacent_node in range(len(self.distance_matrix)):
            if adjacent_node == self.depot_index:
                continue

            edge = (adjacent_node, node_index)
            var = self._add_variable(edge)
            edges_vars.append(var)

        return edges_vars

    def _add_variable(self, edge: Tuple[int, int]) -> GEKKO.Var:
        var = self._edge_chosen_vars.get(edge)

        if var is None:
            var = self._model.Var(value=0, lb=0, ub=1, integer=True)
            self._edge_chosen_vars[edge] = var

        return var

    def _add_constraints(self):
        """
        Add global constraints to the solver.
        """
        for constraint in self._constraints:
            constraint.apply_constraint()

    def _add_objective(self):
        intermedias_sum = 0
        for edge, variable in self._edge_chosen_vars.items():
            duration = self.get_distance(edge)
            intermedias_sum += self._model.Intermediate(duration * variable)

        self._model.Obj(intermedias_sum)

    def _add_options(self):
        # The SOLVER option specifies the type of solver that solves the
        # VRP problem. More on solver options and other parameters can be found on
        # https://gekko.readthedocs.io/en/latest/global.html
        self._model.options.SOLVER = 1


class VRPCPRouteValidation:
    """
    Route validator for constraint programming since a route can have self-loops between vertexes with no depot.
    """

    def __init__(
        self,
        individual_vrp_paths: List[VRPPath],
        distance_matrix: np.ndarray,
        depot_index: int,
    ):
        self.individual_vrp_paths = individual_vrp_paths
        self.distance_matrix = distance_matrix
        self.depot_index = depot_index

    def create_routes(self) -> List[VRPRoute]:
        """
        Creates a list of valid VRP Routes.
        """
        routes = self._construct_routes()
        routes = self._correct_looping_routes(routes)

        return routes

    def _construct_routes(self) -> List[VRPRoute]:
        """
        Constructs Routes from VRP Paths. No validation hapenning here, only getting the initial states of routes.
        """
        path_dict = defaultdict(list)
        for path in self.individual_vrp_paths:
            path_dict[path.from_vertex].append(path.to_vertex)

        routes: List[VRPRoute] = []

        while len(path_dict[self.depot_index]):
            routes.append(self._get_route(path_dict, self.depot_index))

        for starting_point in path_dict.keys():
            if len(path_dict[starting_point]):
                routes.append(self._get_route(path_dict, starting_point))

        return routes

    def _get_route(self, path_dict: Dict[int, List[int]], start_index: int) -> VRPRoute:
        """
        Constructs a particular route by following ending nodes.
        """
        route: List[int] = [start_index]

        while True:
            source = route[-1]

            if not len(path_dict[source]):
                break
            if len(route) > 1 and route[0] == route[-1]:
                break

            next_one = path_dict[source][0]
            path_dict[source] = path_dict[source][1:]
            route.append(next_one)

        return VRPRoute(route)

    def _correct_looping_routes(self, routes: List[VRPRoute]) -> List[VRPRoute]:
        """
        Fixes routes without a depot to be appended to the distance with smallest length.
        """
        routes_without_depot = [x for x in routes if x[0] != self.depot_index]

        for route_without_depot in routes_without_depot:
            routes_from_depot_with_lengths = sorted(
                [
                    (x, get_route_length(x, self.distance_matrix))
                    for x in routes
                    if x[0] == self.depot_index
                ],
                key=lambda x: x[1],
            )
            self._merge_routes(
                routes_from_depot_with_lengths[0][0], route_without_depot
            )

        new_routes = [x for x in routes if x[0] == self.depot_index]

        return new_routes

    def _merge_routes(self, route_with_depot: VRPRoute, route_without_depot: VRPRoute):
        """
        Merges a route with and without a depot to a unique route with a depot.
        """
        route_with_depot.paths = route_with_depot.paths[:-1]
        for index in range(len(route_without_depot) - 1):
            route_with_depot.paths.append(route_without_depot[index])

        route_with_depot.paths.append(route_with_depot[0])


class VRPConstraint(ABC):
    def __init__(self, model: GEKKO):
        self._model = model

    @abstractmethod
    def apply_constraint(self):
        pass


class TimeIncreasesWithPassingFromOneNodeToAnotherConstraint(VRPConstraint):
    """
    Allow progression in time when passing from one node to another.
    """

    def __init__(self, model: GEKKO, variables, time_vars, distance_matrix: np.array):
        super().__init__(model)

        self._variables = variables
        self._time_variables = time_vars
        self._distance_matrix = distance_matrix

    def apply_constraint(self):
        for edge in self._variables:
            (from_node, to_node) = edge
            if from_node < 0 or to_node < 0:
                continue

            self._model.Equation(
                (
                    self._time_variables[from_node]
                    + self._distance_matrix[from_node][to_node]
                )
                * self._variables[edge]
                <= self._time_variables[to_node]
            )


class No3NodeCyclesConstraint(VRPConstraint):
    """
    Do not allow 3 node loops
    """

    def __init__(self, model: GEKKO, variables, node_ids: List[int]):
        super().__init__(model)

        self._variables = variables
        self._node_ids = node_ids

    def apply_constraint(self):
        """
        Do not allow 3 node loops
        """
        for a in self._node_ids:
            for b in self._node_ids:
                if a == b:
                    continue
                for c in self._node_ids:
                    if c == a or c == b:
                        continue
                    self._model.Equation(
                        self._variables[(a, b)]
                        + self._variables[(b, c)]
                        + self._variables[(c, a)]
                        <= 2
                    )


class StartInSourceNodeConstraint(VRPConstraint):
    """
    Whatever the source node is, all of the vehicles must be found in it at some point.
    """

    def __init__(
        self,
        model: GEKKO,
        variables,
        node_ids: List[int],
        no_vehicles: int,
        source_id: int,
    ):
        super().__init__(model)

        self._variables = variables
        self._node_ids = node_ids
        self._no_vehicles = no_vehicles
        self._source_id = source_id

    def apply_constraint(self):
        self._model.Equation(
            sum(self._variables[(self._source_id, n)] for n in self._node_ids)
            == self._no_vehicles
        )


class EndInSinkNodeConstraint(VRPConstraint):
    """
    Whatever the sink node is, all of the vehicles must be found in it at some point.
    """

    def __init__(
        self,
        model: GEKKO,
        variables,
        node_ids: List[int],
        no_vehicles: int,
        sink_id: int,
    ):
        super().__init__(model)

        self._variables = variables
        self._node_ids = node_ids
        self._no_vehicles = no_vehicles
        self._sink_id = sink_id

    def apply_constraint(self):
        self._model.Equation(
            sum(self._variables[(n, self._sink_id)] for n in self._node_ids)
            == self._no_vehicles
        )


class MaximumEdgesActivatedConstraint(VRPConstraint):
    """
    Add total number of paths (edges) that needs to be present.
    """

    def __init__(
        self,
        model: GEKKO,
        variables,
        node_ids: List[int],
        no_vehicles: int,
    ):
        super().__init__(model)

        self._variables = variables
        self._node_ids = node_ids
        self._no_vehicles = no_vehicles

    def apply_constraint(self):
        self._model.Equation(
            sum(self._variables.values()) == len(self._node_ids) + self._no_vehicles
        )


class NoBacktrackingConstraint(VRPConstraint):
    """
    Add no backtracking from one node to another.
    """

    def __init__(
        self,
        model: GEKKO,
        variables,
    ):
        super().__init__(model)
        self._variables = variables

    def apply_constraint(self):
        for edge in self._variables:
            (from_node, to_node) = edge
            if from_node < 0 or to_node < 0:
                continue

            self._model.Equation(
                self._variables[(from_node, to_node)]
                + self._variables[(to_node, from_node)]
                <= 1
            )
