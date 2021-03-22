MERGE (a:P {id: "p1"}) MERGE (b:S {id: "s1"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:S {id: "s2"}) MERGE (b:P {id: "p2"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:S {id: "s1"}) MERGE (b:S {id: "s2"}) CREATE (a)-[e:Friend]->(b);