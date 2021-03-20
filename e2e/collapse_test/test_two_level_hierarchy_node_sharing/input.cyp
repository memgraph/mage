MERGE (a:P {id: "p1"}) MERGE (b:S {id: "s1"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:P {id: "p2"}) MERGE (b:S {id: "s2"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:S {id: "s1"}) MERGE (b:N {id: "n1"}) CREATE (a)-[e:Next]->(b);
MERGE (a:S {id: "s2"}) MERGE (b:N {id: "n1"}) CREATE (a)-[e:Next]->(b);