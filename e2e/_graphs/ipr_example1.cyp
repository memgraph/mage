MERGE (a {id:"A"}) MERGE (b {id: "B"}) CREATE (a)-[:Links]->(b)
MERGE (a {id:"A"}) MERGE (c {id: "C"}) CREATE (a)-[:Links]->(c)
MERGE (c {id:"C"}) MERGE (a {id: "A"}) CREATE (c)-[:Links]->(a)
MERGE (b {id:"B"}) MERGE (c {id: "C"}) CREATE (b)-[:Links]->(c)
