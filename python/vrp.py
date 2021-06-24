from mage.geography.vehicle_routing import DynamicRouting, VRPResult
from mage.geography import (
    create_distance_matrix,
    LATITUDE,
    LONGITUDE,
)
from mage.constraint_programming import VRPConstraintProgrammingSolver

from typing import Dict, List, Tuple

import mgp

__routes = None


class DepotUnspecifiedException(Exception):
    pass


def get_distance_matrix(vertices):
    """
    Assigns distance matrix global object or returns if its already there.
    """

    vertex_positions: List[Dict[str, float]] = []
    for vertex in vertices:
        vertex_positions.append(
            {
                LATITUDE: vertex.properties.get(LATITUDE),
                LONGITUDE: vertex.properties.get(LONGITUDE),
            }
        )

    distance_matrix = create_distance_matrix(vertex_positions)

    return distance_matrix


def get_depot_index(vertices: mgp.Vertices, depot_node: mgp.Vertex):
    """
    Assigns depot index global variable or returns if its already there.
    """
    for idx, vertex in enumerate(vertices):
        if vertex == depot_node:
            return idx

    raise DepotUnspecifiedException("No depot location specified!")


def update_routes(vrp_result: VRPResult):
    global __routes
    __routes = vrp_result


def create_output_from_result(result: VRPResult) -> List[Tuple[int, int, int]]:
    """
    Creates output consumed by the procedure.
    """
    return [
        (route[i], route[i + 1], vehicle_number + 1)
        for (vehicle_number, route) in enumerate(result.routes)
        for i in range(len(route) - 1)
    ]


def get_location_vrp_id(vertices: mgp.Vertices, vertex: mgp.Vertex):
    for (idx, v) in enumerate(vertices):
        if v == vertex:
            return idx


@mgp.read_proc
def route(
    context: mgp.ProcCtx,
    depot_node: mgp.Vertex,
    number_of_vehicles: mgp.Nullable[int] = None,
) -> mgp.Record(
    from_vertex=mgp.Vertex, to_vertex=mgp.Vertex, vehicle_id=mgp.Nullable[int]
):
    """
    The VRP routing returns 2 fields.
        * `from_vertex` represents the starting nodes out of all selected routes (edges) in the complete graph
        * `to_vertex` represents the ending nodes out of all selected routes (edges) in the complete graph

    The input arguments are:
        * `number_of_vehicle` represents the cardinality of fleet with which the problem is going to be solved
        * `depot_label` represents the name of the label which contains the depot node
    """

    if number_of_vehicles is None:
        number_of_vehicles = 1
    if number_of_vehicles <= 0:
        raise Exception("Number of vehicles must be greater than 0.")

    vertices = [v for v in context.graph.vertices]
    distance_matrix = get_distance_matrix(vertices)
    depot_index = get_depot_index(vertices, depot_node)

    solver = VRPConstraintProgrammingSolver(
        number_of_vehicles, distance_matrix, depot_index
    )
    solver.solve()

    result = solver.get_result()

    output = create_output_from_result(result)

    update_routes(result)

    return [
        mgp.Record(
            from_vertex=vertices[o[0]], to_vertex=vertices[o[1]], vehicle_id=o[2]
        )
        for o in output
    ]


@mgp.read_proc
def re_route(
    context: mgp.ProcCtx,
    new_locations: List[mgp.Vertex],
) -> mgp.Record(
    from_vertex=mgp.Vertex, to_vertex=mgp.Vertex, vehicle_id=mgp.Nullable[int]
):
    global __routes
    if __routes is None:
        return
        
    vertices = [v for v in context.graph.vertices]
    distance_matrix = get_distance_matrix(vertices)

    dynamical_router = DynamicRouting(distance_matrix)
    routes = __routes

    for new_location in new_locations:
        routes = dynamical_router.reroute(routes, get_location_vrp_id(vertices, new_location))

    __routes = routes
    
    output = create_output_from_result(routes)

    update_routes(routes)

    return [
        mgp.Record(
            from_vertex=vertices[o[0]], to_vertex=vertices[o[1]], vehicle_id=o[2]
        )
        for o in output
    ]
