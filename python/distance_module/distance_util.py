import math

KM_MULTIPLIER = 0.001
LATITUDE = 'lat'
LONGITUDE = 'lng'


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
        return 0

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