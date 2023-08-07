setup: |-
    CALL meta.reset();
    CREATE TRIGGER meta_stats BEFORE COMMIT EXECUTE CALL meta.update(createdObjects, deletedObjects, removedVertexProperties, removedEdgeProperties, setVertexLabels, removedVertexLabels) YIELD *;

queries:
    - |-
        MERGE (a:Node {id: 0}) MERGE (b:Node {id: 1}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 1}) MERGE (b:Node {id: 2}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 2}) MERGE (b:Node {id: 0}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 3}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 3}) MERGE (b:Node {id: 4}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 3}) MERGE (b:Node {id: 5}) CREATE (a)-[:RELATION]->(b);

cleanup: |-
    CALL meta.reset();
    DROP TRIGGER meta_stats;
