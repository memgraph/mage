CREATE (station1:Station1 {id: 1, name: "Station 1"})
CREATE (station2:Station2 {id: 2, name: "Station 3"})
CREATE (station1)-[j:JOURNEY {id: 1, arrival: "0802", departure: "0803"}]->(station2)
