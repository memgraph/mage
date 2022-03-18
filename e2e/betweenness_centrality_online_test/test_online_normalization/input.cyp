queries:
    - |-
        MERGE (a: Node {id: 30}) MERGE (b: Node {id: 31}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 30}) MERGE (b: Node {id: 32}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 31}) MERGE (b: Node {id: 32}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 32}) MERGE (b: Node {id: 33}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 33}) MERGE (b: Node {id: 34}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 33}) MERGE (b: Node {id: 35}) CREATE (a)-[:RELATION]->(b);
        MERGE (a: Node {id: 34}) MERGE (b: Node {id: 35}) CREATE (a)-[:RELATION]->(b);
    - |-
        MATCH (n: Node) RETURN n;

cleanup: |-
    MATCH (n: Node) DETACH DELETE n;
    CALL betweenness_centrality_online.reset() YIELD *;
    CALL mg.load('betweenness_centrality_online') YIELD *;
