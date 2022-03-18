queries:
    - |-
        MERGE (a: Node {id: 40}) MERGE (b: Node {id: 41}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 40}) MERGE (b: Node {id: 42}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 41}) MERGE (b: Node {id: 42}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 42}) MERGE (b: Node {id: 43}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 43}) MERGE (b: Node {id: 44}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 43}) MERGE (b: Node {id: 45}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 44}) MERGE (b: Node {id: 45}) CREATE (a)-[:RELATION]->(b);

cleanup: |-
    MATCH (n: Node) DETACH DELETE n;
    CALL betweenness_centrality_online.reset() YIELD *;
    CALL mg.load('betweenness_centrality_online') YIELD *;
