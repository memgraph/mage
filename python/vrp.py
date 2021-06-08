from abc import ABC
from gekko import GEKKO
from collections import namedtuple
import mgp

SOURCE_INDEX = -1
SINK_INDEX = -2


@mgp.read_proc
def route(
    context: mgp.ProcCtx,
    number_of_vehicles: mgp.Nullable[int] = None,
    depot_label: mgp.Nullable[str] = "Depot",
    location_label: mgp.Nullable[str] = "Location",
) -> mgp.Record(from_vertex=int, to_vertex=int, value=str):
    if number_of_vehicles is None:
        number_of_vehicles = 1
    if number_of_vehicles <= 0:
        raise Exception("Number of vehicles must be greater than 0.")

    m = GEKKO(remote=False)
    solver = VRPProblemSolver(number_of_vehicles, context.graph.vertices, m, depot_label, location_label)
    solver.solve()

    # Results
    print("\nResults")
    print(f"Obj={m.options.objfcnval}")
    print_time_vars(solver.time_vars)
    # print_edge_chosen_vars(solver.edge_chosen_vars)

    return [
        mgp.Record(
            from_vertex=key[0] if key[0] >= 0 else solver._depot_id,
            to_vertex=key[1] if key[1] >= 0 else solver._depot_id,
            value=str(var.value),
        )
        for key, var in solver.edge_chosen_vars.items()
        if int(var.value[0]) == 1
    ]


class VRPProblemSolver:
    def __init__(self, no_vehicles, vertices: mgp.Vertices, model: GEKKO, depot_label, location_label):
        self.no_vehicles = no_vehicles
        self.depot_label = depot_label
        self.location_label = location_label

        self._model = model
        self._nodes_ids = list()
        self._depot_id = None
        self._edge_durations = dict()

        self.edge_chosen_vars = dict()
        self.time_vars = dict()

        self._initialize(vertices)

        self._add_constraints()

        self._add_objective()
        self._add_options()

    def solve(self):
        self._model.solve()

    def _initialize(self, vertices: mgp.Vertices):
        for node in vertices:
            if self.depot_label in node.labels:
                self._initialize_depot_node(node)

        for node in vertices:
            if self.location_label in node.labels:
                self._initialize_location_node(node)

    def _add_constraints(self):
        visited_pairs = dict()
        for edge in self.edge_chosen_vars:
            (from_node, to_node) = edge
            min_node, max_node = min(from_node, to_node), max(from_node, to_node)
            if min_node < 0 or max_node < 0:
                continue

            self._model.Equation(
                (self.time_vars[from_node] + self._edge_durations[edge]) * self.edge_chosen_vars[edge]
                < self.time_vars[to_node]
            )

            if (min_node, max_node) in visited_pairs:
                continue
            visited_pairs[(min_node, max_node)] = True

            self._add_no_backtracking_constraint(from_node, to_node)

        self._add_start_in_source_node()
        self._add_end_in_sink_node()
        self._add_no_3_node_cycle_constraint()

    def _add_objective(self):
        intermedias = []
        for edge, variable in self.edge_chosen_vars.items():
            duration = self._edge_durations.get(edge)
            mul = self._model.Intermediate(duration * variable)
            intermedias.append(mul)

        self._model.Obj(sum(intermedias))

    def _add_options(self):
        self._model.options.SOLVER = 1

    def _add_no_backtracking_constraint(self, from_node, to_node):
        self._model.Equation(
            self.edge_chosen_vars[(from_node, to_node)] + self.edge_chosen_vars[(to_node, from_node)] <= 1
        )

    def _add_no_3_node_cycle_constraint(self):
        for a in self._nodes_ids:
            for b in self._nodes_ids:
                if a == b:
                    continue
                for c in self._nodes_ids:
                    if c == a or c == b:
                        continue
                    self._model.Equation(
                        self.edge_chosen_vars[(a, b)] + self.edge_chosen_vars[(b, c)] + self.edge_chosen_vars[(c, a)]
                        <= 2
                    )

    def _add_start_in_source_node(self):
        self._model.Equation(sum(self.edge_chosen_vars[(SOURCE_INDEX, n)] for n in self._nodes_ids) == self.no_vehicles)

    def _add_end_in_sink_node(self):
        self._model.Equation(sum(self.edge_chosen_vars[(n, SINK_INDEX)] for n in self._nodes_ids) == self.no_vehicles)

    def _initialize_depot_node(self, node: mgp.Vertex):
        self._depot_id = node.id

    def _initialize_location_node(self, node: mgp.Vertex):
        self._nodes_ids.append(node.id)

        self.time_vars[node.id] = self._model.Var(value=0, lb=0, integer=False)

        # Initialize starting point and sinking point for every vehicle
        self._add_variable((SOURCE_INDEX, node.id), 0)
        self._add_variable((node.id, SINK_INDEX), 0)

        # For every node, draw lengths from and to it, with duration of edges
        out_vars = self._add_variables(node.out_edges)
        in_vars = self._add_variables(node.in_edges)

        # Either it was a beginning node, or a vehicle has visited it in the drive.
        if len(out_vars) > 0:
            self._model.Equation(self.edge_chosen_vars[(node.id, SINK_INDEX)] + sum(out_vars) == 1)

        if len(in_vars) > 0:
            self._model.Equation(self.edge_chosen_vars[(SOURCE_INDEX, node.id)] + sum(in_vars) == 1)

    def _add_variable(self, edge, duration):
        var = self.edge_chosen_vars.get(edge)

        if var is None:
            var = self._model.Var(value=0, lb=0, ub=1, integer=True)

            self.edge_chosen_vars[edge] = var
            self._edge_durations[edge] = self._model.Const(value=duration)

        return var

    def _add_variables(self, edges):
        edges_vars = []
        for edge in edges:
            from_node = edge.from_vertex.id
            to_node = edge.to_vertex.id
            if from_node == self._depot_id or to_node == self._depot_id:
                continue

            duration = edge.properties.get("duration")
            if duration is None:
                raise Exception(f"Edge (id: {from_node})->(id: {to_node}) does not have duration property!")

            var = self._add_variable((from_node, to_node), duration)
            edges_vars.append(var)

        return edges_vars


def print_time_vars(time_vars):
    for time, variable in time_vars.items():
        print(f"{time}={variable.value}")


def print_edge_chosen_vars(edge_chosen_vars):
    for edge, variable in edge_chosen_vars.items():
        print(f"{edge}={variable.value}")
