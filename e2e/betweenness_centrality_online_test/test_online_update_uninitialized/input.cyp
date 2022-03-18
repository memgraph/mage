setup: |-
        CREATE TRIGGER test_create_edge BEFORE COMMIT EXECUTE CALL betweenness_centrality_online.update(createdVertices, createdEdges, deletedVertices, deletedEdges) YIELD *;

queries:
    - |-
        MERGE (a: Node {id: 10}) MERGE (b: Node {id: 11}) CREATE (a)-[:RELATION]->(b);

cleanup: |-
    DROP TRIGGER test_create_edge;
    MATCH (n: Node) DETACH DELETE n;
    CALL betweenness_centrality_online.reset() YIELD *;
    CALL mg.load('betweenness_centrality_online') YIELD *;
