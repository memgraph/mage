setup: |-
  CALL meta.reset();
  CREATE TRIGGER meta_stats BEFORE COMMIT EXECUTE CALL meta.update(createdObjects, deletedObjects, removedVertexProperties, removedEdgeProperties, setVertexLabels, removedVertexLabels) YIELD *;

queries:
  - |-
    MERGE (a:node {id: 0}) MERGE (b:node {id: 1}) CREATE (a)-[:RELATION {koza: true}]->(b);
    MERGE (a:node {id: 1}) MERGE (b:node {id: 2}) CREATE (a)-[:RELATION {viroza: true}]->(b);
    MERGE (a:node {id: 2}) MERGE (b:node {id: 0}) CREATE (a)-[:RELATION {mimoza: true}]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION {ciroza: false}]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 4}) CREATE (a)-[:RELATION {prognoza: "cold"}]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 5}) CREATE (a)-[:RELATION]->(b);
    MATCH (:Node)-[r:RELATION]-(:Node) DELETE r;
  - |-
    MATCH (n:Node {id: 3}) DETACH DELETE n;
cleanup: |-
  CALL meta.reset();
  DROP TRIGGER meta_stats;
