from collections import namedtuple

Depot = namedtuple("Depot", ["lat", "lng"])
Location = namedtuple("Location", ["lat", "lng"])


locations = [
    Location(45.81397494712325, 15.977107314009686),
    Location(45.809786288641924, 15.969953021143715),
    Location(45.801513169575195, 15.979868413090431),
    Location(45.80062044456095, 15.971453134506456),
    Location(45.80443233736649, 15.993114737391515),
    Location(45.77165828306254, 15.943635971437576),
    Location(45.785275159565806, 15.947448603375522),
    Location(45.780581597098646, 15.935278141510148),
    Location(45.82208303601525, 16.019498047049822),
]

depot = Depot(45.7872369074369, 15.984469921454693)


def main():
    print("MATCH (n) DETACH DELETE n;")

    for i in range(len(locations)):
        query = (
            f"CREATE (:Location {{lat:{locations[i].lat}, lng:{locations[i].lng}}});"
        )
        print(query)

    query = f"CREATE (:Depot {{lat:{depot.lat}, lng:{depot.lng}}});"
    print(query)

    query = f"""
MATCH (d:Depot)
CALL vrp.route(d) YIELD from_vertex, to_vertex, vehicle_id
CREATE (from_vertex)-[r:Route]->(to_vertex);

MATCH (n)-[r:Route]->(m)
RETURN n, r, m;
"""
    print(query)


if __name__ == "__main__":
    main()
