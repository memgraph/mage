import mgp
from distance_util import calculate_distance_between_points

@mgp.read_proc
def single(context: mgp.ProcCtx,
              start: mgp.Nullable[mgp.Any],
              end: mgp.Nullable[mgp.Any],
              metrics: str = 'm'
              ) -> mgp.Record(distance=mgp.Number):
    '''
    This distance calculator procedure  for one entry returns 1 field.

      * `distance` is the final result obtained by calculating distance (in metres) between the 2 points who
        each have its latitude and longitude.

    The procedure can be invoked in openCypher using the following calls:
      CALL distance_calculator.single((:Point {lat:1, lng:2}), (:Point {lat:2, lng:3.5})) YIELD distance;
      MATCH (n1:Point), (n2:Point) CALL distance_calculator.procedure(n, 1) YIELD * RETURN *;
    '''

    if not isinstance(start, (mgp.Vertex)) or not isinstance(end, (mgp.Vertex)):
        return mgp.Record(distance=None)

    distance = calculate_distance_between_points(
        dict(start.properties.items()),
        dict(end.properties.items()),
        metrics)

    return mgp.Record(distance=distance)

@mgp.read_proc
def multiple(context: mgp.ProcCtx,
              start_points: mgp.Nullable[mgp.Any],
              end_points: mgp.Nullable[mgp.Any],
              metrics: str = 'm'
              ) -> mgp.Record(distances=list):
    '''
    This distance calculator procedure  for multiple entries returns 1 field.

      * `distances` is the final result obtained by calculating distances (in metres) between pairs of
      points who each have its latitude and longitude

    The procedure can be invoked in openCypher using the following calls:
      CALL distance_calculator.multiple([(:Point {lat:1, lng:2})]), [(:Point {lat:3,lng:4.5})])) YIELD distances;
    '''

    if len(start_points) != len(end_points) or len(start_points) == 0:
        return mgp.Record(distances=None)

    for i in range(len(start_points)):
        if not isinstance(start_points[i], (mgp.Vertex)) or not isinstance(end_points[i], (mgp.Vertex)):
            return mgp.Record(distances=None)

    distances = []
    for i in range(len(start_points)):
        d = calculate_distance_between_points(
            dict(start_points[i].properties.items()),
            dict(end_points[i].properties.items()),
            metrics)

        distances.append(d)

    return mgp.Record(distances=distances)


if __name__ == '__main__':
    calculate_distance_between_points({'lat':1,'lng':2}, {'lat':2, 'lng':3})