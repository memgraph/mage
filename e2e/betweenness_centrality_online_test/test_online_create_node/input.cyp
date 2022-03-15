setup: |-
        CREATE TRIGGER test_create_node BEFORE COMMIT EXECUTE CALL betweenness_centrality_online.update(createdVertices, createdEdges, deletedVertices, deletedEdges) YIELD *;

queries:
    - |-
        MERGE (a: Node {id: 0}) MERGE (b: Node {id: 1}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 0}) MERGE (b: Node {id: 2}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 1}) MERGE (b: Node {id: 2}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 2}) MERGE (b: Node {id: 3}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 3}) MERGE (b: Node {id: 4}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 3}) MERGE (b: Node {id: 5}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 4}) MERGE (b: Node {id: 5}) CREATE (a)-[:RELATION]->(b);
    - |-
        CREATE (n: Node {id: 6});

cleanup: |-
    DROP TRIGGER test_create_node;
    MATCH (n: Node) DETACH DELETE n;
    CALL betweenness_centrality_online.reset() YIELD *;
    CALL mg.load('betweenness_centrality_online') YIELD *;
