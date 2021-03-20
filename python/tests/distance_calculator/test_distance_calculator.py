import pytest
from mage.geography import calculate_distance_between_points


@pytest.fixture
def points():
    return {"lat": 44.1194, "lng": 15.2314}, {"lat": 45.8150, "lng": 15.9819}


def test_distance_between_points_km(points):
    point_a, point_b = points
    result = calculate_distance_between_points(point_a, point_b, metrics="km")

    assert result == pytest.approx(197.56, 0.1)


def test_distance_between_points_m(points):
    point_a, point_b = points
    result = calculate_distance_between_points(point_a, point_b, metrics="m")

    assert result == pytest.approx(197568.2, 0.1)


def test_wrong_metrics(points):
    point_a, point_b = points
    result = calculate_distance_between_points(point_a, point_b, metrics="r")
    assert result is None


def test_wrong_keys(points):
    point_a, point_b = points
    result = calculate_distance_between_points(point_a, point_b, metrics="r")
    assert result is None
