MERGE (a:P {id: "p1"}) MERGE (b:S {id: "s1"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:P {id: "p2"}) MERGE (b:S {id: "s1"}) CREATE (a)-[e:Edge]->(b);