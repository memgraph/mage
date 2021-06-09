from gekko import GEKKO
from mage.geography import VRPPath, VRPResult, VRPSolver
from typing import List, Tuple
import numpy as np


class VRPConstraintProgrammingSolver(VRPSolver):
    """
    This constraint solver solves the Vehicle Routing Problem with constraint programming using GEKKO.
    """

    SOURCE_INDEX = -1
    SINK_INDEX = -2

    def __init__(self, no_vehicles: int, distance_matrix: np.array, depot_index: int):
        self._model = GEKKO(remote=False)

        self.no_vehicles = no_vehicles
        self.distance_matrix = distance_matrix
        self.depot_index = depot_index

        self._edge_chosen_vars = dict()
        self._time_vars = dict()
        self._location_node_ids = [
            x for x in range(len(distance_matrix)) if x != self.depot_index
        ]

        self._initialize()

        self._add_constraints()
        self._add_objective()
        self._add_options()

    def solve(self):
        self._model.solve()

    def get_result(self) -> VRPResult:
        return VRPResult(
            [
                VRPPath(
                    key[0] if key[0] >= 0 else self.depot_index,
                    key[1] if key[1] >= 0 else self.depot_index,
                )
                for key, var in self._edge_chosen_vars.items()
                if int(var.value[0]) == 1
            ]
        )

    def get_distance(self, edge: Tuple[int, int]) -> float:
        node_from, node_to = edge

        if node_from in [
            VRPConstraintProgrammingSolver.SINK_INDEX,
            VRPConstraintProgrammingSolver.SOURCE_INDEX,
        ] or node_to in [
            VRPConstraintProgrammingSolver.SINK_INDEX,
            VRPConstraintProgrammingSolver.SOURCE_INDEX,
        ]:
            return 0

        return self.distance_matrix[node_from][node_to]

    def print_results(self):
        # Results
        print("\nResults")
        print(f"Obj={self._model.options.objfcnval}")
        self.print_time_vars()

    def _initialize(self):
        for node_index in range(len(self.distance_matrix)):
            if node_index in self._location_node_ids:
                self._initialize_location_node(node_index)

    def _initialize_location_node(self, node_index: int):
        self._time_vars[node_index] = self._model.Var(value=0, lb=0, integer=False)

        # Initialize starting point and sinking point for every vehicle
        self._add_variable((VRPConstraintProgrammingSolver.SOURCE_INDEX, node_index))
        self._add_variable((node_index, VRPConstraintProgrammingSolver.SINK_INDEX))

        # For every node, draw lengths from and to it, with duration of edges
        out_vars = self._add_adjacent_edge_variables(node_index)
        in_vars = self._add_adjacent_edge_variables(node_index, input=True)

        # Either it was a beginning node, or a vehicle has visited it in the drive.
        if len(out_vars) > 0:
            self._model.Equation(
                self._edge_chosen_vars[
                    (node_index, VRPConstraintProgrammingSolver.SINK_INDEX)
                ]
                + sum(out_vars)
                == 1
            )

        if len(in_vars) > 0:
            self._model.Equation(
                self._edge_chosen_vars[
                    (VRPConstraintProgrammingSolver.SOURCE_INDEX, node_index)
                ]
                + sum(in_vars)
                == 1
            )

    def _add_adjacent_edge_variables(
        self, node_index: int, input: bool = False
    ) -> List[Tuple[int, int]]:
        edges_vars = []

        for adjacent_node in range(len(self.distance_matrix)):
            if adjacent_node == self.depot_index:
                continue

            edge = (node_index, adjacent_node)
            if input:
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
        self._add_maximum_edges_activated()

        visited_pairs = dict()
        for edge in self._edge_chosen_vars:
            (from_node, to_node) = edge
            min_node, max_node = min(from_node, to_node), max(from_node, to_node)
            if min_node < 0 or max_node < 0:
                continue

            self._model.Equation(
                (self._time_vars[from_node] + self.distance_matrix[from_node][to_node])
                * self._edge_chosen_vars[edge]
                <= self._time_vars[to_node]
            )

            if (min_node, max_node) in visited_pairs:
                continue
            visited_pairs[(min_node, max_node)] = True

            self._add_no_backtracking_constraint(from_node, to_node)

        self._add_start_in_source_node()
        self._add_end_in_sink_node()
        self._add_no_3_node_cycle_constraint()

    def _add_maximum_edges_activated(self):
        """
        Add total number of paths (edges) that needs to be present.
        """
        self._model.Equation(
            sum(self._edge_chosen_vars.values())
            == len(self._location_node_ids) + self.no_vehicles
        )

    def _add_no_backtracking_constraint(self, from_node: int, to_node: int):
        """
        Add rule that nodes can't go back in paths.
        """
        self._model.Equation(
            self._edge_chosen_vars[(from_node, to_node)]
            + self._edge_chosen_vars[(to_node, from_node)]
            <= 1
        )

    def _add_no_3_node_cycle_constraint(self):
        """
        Do not allow 3 node loops
        """
        for a in self._location_node_ids:
            for b in self._location_node_ids:
                if a == b:
                    continue
                for c in self._location_node_ids:
                    if c == a or c == b:
                        continue
                    self._model.Equation(
                        self._edge_chosen_vars[(a, b)]
                        + self._edge_chosen_vars[(b, c)]
                        + self._edge_chosen_vars[(c, a)]
                        <= 2
                    )

    def _add_start_in_source_node(self):
        self._model.Equation(
            sum(
                self._edge_chosen_vars[(VRPConstraintProgrammingSolver.SOURCE_INDEX, n)]
                for n in self._location_node_ids
            )
            == self.no_vehicles
        )

    def _add_end_in_sink_node(self):
        self._model.Equation(
            sum(
                self._edge_chosen_vars[(n, VRPConstraintProgrammingSolver.SINK_INDEX)]
                for n in self._location_node_ids
            )
            == self.no_vehicles
        )

    def _add_objective(self):
        intermedias = []
        for edge, variable in self._edge_chosen_vars.items():
            duration = self.get_distance(edge)
            mul = self._model.Intermediate(duration * variable)
            intermedias.append(mul)

        self._model.Obj(sum(intermedias))

    def _add_options(self):
        self._model.options.SOLVER = 1

    def print_time_vars(self):
        for time, variable in self._time_vars.items():
            print(f"{time}={variable.value}")

    def print__edge_chosen_vars(self):
        for edge, variable in self._edge_chosen_vars.items():
            print(f"{edge}={variable.value}")
