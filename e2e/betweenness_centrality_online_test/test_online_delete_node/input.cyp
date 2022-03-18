setup: |-
        CREATE TRIGGER test_delete_node BEFORE COMMIT EXECUTE CALL betweenness_centrality_online.update(createdVertices, createdEdges, deletedVertices, deletedEdges) YIELD *;

queries:
    - |-
        MERGE (a: Node {id: 20}) MERGE (b: Node {id: 21}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 20}) MERGE (b: Node {id: 22}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 21}) MERGE (b: Node {id: 22}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 22}) MERGE (b: Node {id: 23}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 23}) MERGE (b: Node {id: 24}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 23}) MERGE (b: Node {id: 25}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 24}) MERGE (b: Node {id: 25}) CREATE (a)-[:RELATION]->(b);
        CREATE (n: Node {id: 26});
    - |-
        MATCH (n: Node {id: 26}) DETACH DELETE n;

cleanup: |-
    DROP TRIGGER test_delete_node;
    MATCH (n: Node) DETACH DELETE n;
    CALL betweenness_centrality_online.reset() YIELD *;
    CALL mg.load('betweenness_centrality_online') YIELD *;
