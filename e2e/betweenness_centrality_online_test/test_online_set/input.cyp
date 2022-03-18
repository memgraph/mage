queries:
    - |-
        MERGE (a: Node {id: 60}) MERGE (b: Node {id: 61}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 60}) MERGE (b: Node {id: 62}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 61}) MERGE (b: Node {id: 62}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 62}) MERGE (b: Node {id: 63}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 63}) MERGE (b: Node {id: 64}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 63}) MERGE (b: Node {id: 65}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 64}) MERGE (b: Node {id: 65}) CREATE (a)-[:RELATION]->(b);

cleanup: |-
    MATCH (n: Node) DETACH DELETE n;
    CALL betweenness_centrality_online.reset() YIELD *;
    CALL mg.load('betweenness_centrality_online') YIELD *;
