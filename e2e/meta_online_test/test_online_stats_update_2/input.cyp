setup: CALL meta.reset();
queries:
  - |-
    MERGE (a:Node {id: 0}) MERGE (b:Node {id: 1}) CREATE (a)-[:Relation]->(b);
    MERGE (a:Node {id: 1}) MERGE (b:Node {id: 2}) CREATE (a)-[:Relation]->(b);
    MERGE (a:Node {id: 2}) MERGE (b:Node {id: 0}) CREATE (a)-[:Relation]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 3}) CREATE (a)-[:Relation]->(b);
  - >-
    CREATE TRIGGER meta_stats BEFORE COMMIT EXECUTE CALL
    meta.update(createdObjects, deletedObjects, removedVertexProperties,
    removedEdgeProperties, setVertexLabels, removedVertexLabels) YIELD *;

    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 4}) CREATE (a)-[:Relation]->(b);

    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 5}) CREATE (a)-[:Relation]->(b);
cleanup: |-
  CALL meta.reset();
  DROP TRIGGER meta_stats;