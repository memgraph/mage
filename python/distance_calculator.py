import mgp
import math
import itertools

KM_MULTIPLIER = 0.001
LATITUDE = 'lat'
LONGITUDE = 'lng'
valid_metrics = ['m', 'km']

@mgp.read_proc
def single(context: mgp.ProcCtx,
              start: mgp.Nullable[mgp.Vertex],
              end: mgp.Nullable[mgp.Vertex],
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

    distance = calculate_distance_between_points(
        dict(start.properties.items()),
        dict(end.properties.items()),
        metrics)

    return mgp.Record(distance=distance)


@mgp.read_proc
def multiple(context: mgp.ProcCtx,
              start_points: list,
              end_points: list,
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

    if not all(isinstance(start, mgp.Vertex) and isinstance(end, mgp.Vertex)
        for start, end in zip(start_points, end_points)):
        return mgp.Record(distance=None)

    distances = []
    for start_point, end_point in zip(start_points, end_points):
        d = calculate_distance_between_points(
            dict(start_point.properties.items()),
            dict(end_point.properties.items()),
            metrics)

        distances.append(d)

    return mgp.Record(distances=distances)


def calculate_distance_between_points(start, end, metrics='m'):
    '''
    Returns distance based on the metrics between 2 points.
    :param start: Start node - dictionary with lat and lng
    :param end: End node - dictionary with lat and lng
    :param metrics: m - in metres, km - in kilometres
    :return: float distance
    '''

    if (LATITUDE not in start.keys()
        or LONGITUDE not in start.keys()
        or LATITUDE not in end.keys()
        or LONGITUDE not in end.keys()):
        return None

    if metrics.lower() not in valid_metrics:
        return None

    lat_1 = start[LATITUDE]
    lng_1 = start[LONGITUDE]
    lat_2 = end[LATITUDE]
    lng_2 = end[LONGITUDE]

    R = 6371E3
    pi_radians = math.pi / 180.00

    phi_1 = lat_1 * pi_radians
    phi_2 = lat_2 * pi_radians
    delta_phi = (lat_2 - lat_1) * pi_radians
    delta_lambda = (lng_2 - lng_1) * pi_radians

    sin_delta_phi = math.sin(delta_phi / 2.0)
    sin_delta_lambda = math.sin(delta_lambda / 2.0)

    a = sin_delta_phi ** 2 + math.cos(phi_1) * math.cos(phi_2) * (sin_delta_lambda ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    #Distance in metres
    distance = R * c

    if metrics.lower() == 'km':
        distance *= KM_MULTIPLIER

    return distance
