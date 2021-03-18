MERGE (a:C {id: "c"}) MERGE (b:P {id: "p1"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:C {id: "c"}) MERGE (b:P {id: "p2"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:P {id: "p1"}) MERGE (b:S {id: "s1"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:P {id: "p1"}) MERGE (b:S {id: "s2"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:P {id: "p2"}) MERGE (b:S {id: "s3"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:P {id: "p2"}) MERGE (b:S {id: "s4"}) CREATE (a)-[e:Edge]->(b);
MERGE (a:S {id: "s3"}) MERGE (b:S {id: "s2"}) CREATE (a)-[e:Friend]->(b);