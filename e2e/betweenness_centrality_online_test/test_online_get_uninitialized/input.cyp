queries:
    - |-
        MERGE (a: Node {id: 50}) MERGE (b: Node {id: 51}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 50}) MERGE (b: Node {id: 52}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 51}) MERGE (b: Node {id: 52}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 52}) MERGE (b: Node {id: 53}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 53}) MERGE (b: Node {id: 54}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 53}) MERGE (b: Node {id: 55}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 54}) MERGE (b: Node {id: 55}) CREATE (a)-[:RELATION]->(b);

cleanup: |-
    MATCH (n: Node) DETACH DELETE n;
    CALL betweenness_centrality_online.reset() YIELD *;
    CALL mg.load('betweenness_centrality_online') YIELD *;
