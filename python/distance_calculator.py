import mgp
import itertools
from mage.distance_calculator import calculate_distance_between_points
from typing import List


@mgp.read_proc
def single(
    context: mgp.ProcCtx,
    start: mgp.Nullable[mgp.Vertex],
    end: mgp.Nullable[mgp.Vertex],
    metrics: str = "m",
) -> mgp.Record(distance=mgp.Number):
    """
    This distance calculator procedure  for one entry returns 1 field.

      * `distance` is the final result obtained by calculating distance (in metres) between the 2 points who
        each have its latitude and longitude.

    The procedure can be invoked in openCypher using the following calls:
      CALL distance_calculator.single((:Point {lat:1, lng:2}), (:Point {lat:2, lng:3.5})) YIELD distance;
      MATCH (n1:Point), (n2:Point) CALL distance_calculator.procedure(n, 1) YIELD * RETURN *;
    """

    distance = calculate_distance_between_points(
        dict(start.properties.items()), dict(end.properties.items()), metrics
    )

    return mgp.Record(distance=distance)


@mgp.read_proc
def multiple(
    context: mgp.ProcCtx,
    start_points: List[mgp.Vertex],
    end_points: List[mgp.Vertex],
    metrics: str = "m",
) -> mgp.Record(distances=List[float]):
    """
    This distance calculator procedure  for multiple entries returns 1 field.

      * `distances` is the final result obtained by calculating distances (in metres) between pairs of
      points who each have its latitude and longitude

    The procedure can be invoked in openCypher using the following calls:
      CALL distance_calculator.multiple([(:Point {lat:1, lng:2})]), [(:Point {lat:3,lng:4.5})])) YIELD distances;
    """

    if len(start_points) != len(end_points) or len(start_points) == 0:
        return mgp.Record(distances=None)

    if not all(
        isinstance(point, mgp.Vertex)
        for point in itertools.chain(start_points, end_points)
    ):
        return mgp.Record(distance=None)

    distances = []
    for start_point, end_point in zip(start_points, end_points):
        d = calculate_distance_between_points(
            dict(start_point.properties.items()),
            dict(end_point.properties.items()),
            metrics,
        )

        distances.append(d)

    return mgp.Record(distances=distances)
