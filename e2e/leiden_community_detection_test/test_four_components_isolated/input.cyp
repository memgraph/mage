MERGE (a:Node {id: 0})
MERGE (b:Node {id: 1})
MERGE (c:Node {id: 3})
CREATE (a)-[:RELATION]->(b),
       (a)-[:RELATION]->(c),
       (b)-[:RELATION]->(c);
MERGE (d:Node {id: 4});
MERGE (e:Node {id: 5});