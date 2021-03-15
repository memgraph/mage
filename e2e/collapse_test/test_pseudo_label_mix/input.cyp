MERGE (a:P {id: "p1"}) MERGE (b:S {id: "s1"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:P {id: "p2"}) MERGE (b:S {id: "s2"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:P {id: "p3"}) MERGE (b:S {id: "s3"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:S {id: "s1"}) MERGE (b:S {id: "s2"}) CREATE (a)-[e:Friend]->(b);
MERGE (a:S {id: "s1"}) MERGE (b:T {id: "t"}) CREATE (a)-[e:Transport]->(b);
MERGE (a:T {id: "t"}) MERGE (b:S {id: "s3"}) CREATE (a)-[e:Transport]->(b);