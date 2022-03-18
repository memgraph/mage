setup: |-
        CREATE TRIGGER test_delete_edge BEFORE COMMIT EXECUTE CALL betweenness_centrality_online.update(createdVertices, createdEdges, deletedVertices, deletedEdges) YIELD *;

queries:
    - |-
        MERGE (a: Node {id: 10}) MERGE (b: Node {id: 11}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 10}) MERGE (b: Node {id: 12}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 11}) MERGE (b: Node {id: 12}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 12}) MERGE (b: Node {id: 13}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 13}) MERGE (b: Node {id: 14}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 13}) MERGE (b: Node {id: 15}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 14}) MERGE (b: Node {id: 15}) CREATE (a)-[:RELATION]->(b);
    - |-
        MATCH (a: Node {id: 14})-[r:RELATION]->(b: Node {id: 15}) DELETE r;

cleanup: |-
    DROP TRIGGER test_delete_edge;
    MATCH (n: Node) DETACH DELETE n;
    CALL betweenness_centrality_online.reset() YIELD *;
    CALL mg.load('betweenness_centrality_online') YIELD *;
