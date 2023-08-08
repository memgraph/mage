MERGE (a:Person {name: "Phoebe"}) MERGE (b:Person {name: "Joey"}) CREATE (a)-[f:FRIENDS]->(b);
MERGE (a:Person {name: "Phoebe"}) MERGE (b:Person {name: "Joey"}) CREATE (b)-[f:FRIENDS]->(a);
