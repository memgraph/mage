from mage.geography.vehicle_routing import InvalidRouteException, VRPRoute
import pytest
import numpy as np

from mage.geography import create_distance_matrix, InvalidDepotException
from mage.constraint_programming.vrp_cp_solver import (
    VRPConstraintProgrammingSolver,
    VRPPath,
    VRPCPRouteValidation,
)


@pytest.fixture
def default_distance_matrix():
    return np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])


@pytest.fixture
def locations_size_10():
    locations = [
        {"lat": 45.81397494712325, "lng": 15.977107314009686},
        {"lat": 45.809786288641924, "lng": 15.969953021143715},
        {"lat": 45.801513169575195, "lng": 15.979868413090431},
        {"lat": 45.80062044456095, "lng": 15.971453134506456},
        {"lat": 45.80443233736649, "lng": 15.993114737391515},
        {"lat": 45.77165828306254, "lng": 15.943635971437576},
        {"lat": 45.785275159565806, "lng": 15.947448603375522},
        {"lat": 45.780581597098646, "lng": 15.935278141510148},
        {"lat": 45.82208303601525, "lng": 16.019498047049822},
        {"lat": 45.7872369074369, "lng": 15.984469921454693},
    ]

    return locations


locations = [
    {"lat": 45.81397494712325, "lng": 15.977107314009686},
    {"lat": 45.809786288641924, "lng": 15.969953021143715},
    {"lat": 45.801513169575195, "lng": 15.979868413090431},
    {"lat": 45.80062044456095, "lng": 15.971453134506456},
    {"lat": 45.80443233736649, "lng": 15.993114737391515},
    {"lat": 45.77165828306254, "lng": 15.943635971437576},
    {"lat": 45.785275159565806, "lng": 15.947448603375522},
    {"lat": 45.780581597098646, "lng": 15.935278141510148},
    {"lat": 45.82208303601525, "lng": 16.019498047049822},
    {"lat": 45.7872369074369, "lng": 15.984469921454693},
]


def test_negative_depot_index_raise_exception(default_distance_matrix):
    with pytest.raises(InvalidDepotException):
        VRPConstraintProgrammingSolver(
            no_vehicles=2, distance_matrix=default_distance_matrix, depot_index=-1
        )


def test_depot_index_to_big_raise_exception(default_distance_matrix):
    with pytest.raises(InvalidDepotException):
        VRPConstraintProgrammingSolver(
            no_vehicles=2,
            distance_matrix=default_distance_matrix,
            depot_index=len(default_distance_matrix),
        )


def test_merge_routes_without_corresponding_depot(locations_size_10):
    distance_matrix = create_distance_matrix(locations_size_10)

    vrp_paths = [
        VRPPath(9, 6),
        VRPPath(6, 7),
        VRPPath(7, 5),
        VRPPath(5, 9),
        VRPPath(9, 8),
        VRPPath(8, 9),
        VRPPath(0, 4),
        VRPPath(4, 2),
        VRPPath(2, 3),
        VRPPath(3, 1),
        VRPPath(1, 0),
    ]

    expected_routes = [VRPRoute([9, 6, 7, 5, 0, 4, 2, 3, 1, 9]), VRPRoute([9, 8, 9])]

    routes = VRPCPRouteValidation(
        vrp_paths, distance_matrix, len(distance_matrix) - 1
    ).create_routes()

    assert len(routes) == 2
    assert routes == expected_routes


def test_all_routes_have_corresponding_depot(locations_size_10):
    distance_matrix = create_distance_matrix(locations_size_10)

    vrp_paths = [
        VRPPath(9, 6),
        VRPPath(6, 7),
        VRPPath(7, 5),
        VRPPath(5, 9),
        VRPPath(9, 8),
        VRPPath(8, 0),
        VRPPath(0, 4),
        VRPPath(4, 2),
        VRPPath(2, 3),
        VRPPath(3, 1),
        VRPPath(1, 9),
    ]

    expected_routes = [VRPRoute([9, 6, 7, 5, 9]), VRPRoute([9, 8, 0, 4, 2, 3, 1, 9])]

    routes = VRPCPRouteValidation(
        vrp_paths, distance_matrix, len(distance_matrix) - 1
    ).create_routes()

    assert len(routes) == 2
    assert routes == expected_routes


def test_route_of_length_2_raises_error():
    with pytest.raises(InvalidRouteException):
        VRPRoute([9, 9])


def test_route_of_not_returning_to_beginning_location_raises_error():
    with pytest.raises(InvalidRouteException):
        VRPRoute([1, 2, 3, 4, 5])
