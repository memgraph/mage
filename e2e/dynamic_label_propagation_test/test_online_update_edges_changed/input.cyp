setup: |-
    CREATE TRIGGER testing AFTER COMMIT EXECUTE CALL dynamic_label_propagation.update(createdVertices, createdEdges, updatedVertices, updatedEdges, deletedVertices, deletedEdges);

queries:
    - |-
        MERGE (a:Node {id: 0}) MERGE (b:Node {id: 1}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 1}) MERGE (b:Node {id: 0}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 0}) MERGE (b:Node {id: 2}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 2}) MERGE (b:Node {id: 0}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 0}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 3}) MERGE (b:Node {id: 0}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 1}) MERGE (b:Node {id: 2}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 2}) MERGE (b:Node {id: 1}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 1}) MERGE (b:Node {id: 4}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 4}) MERGE (b:Node {id: 1}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 2}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 3}) MERGE (b:Node {id: 2}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 2}) MERGE (b:Node {id: 9}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 9}) MERGE (b:Node {id: 2}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 3}) MERGE (b:Node {id: 13}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 13}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 4}) MERGE (b:Node {id: 5}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 5}) MERGE (b:Node {id: 4}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 4}) MERGE (b:Node {id: 6}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 6}) MERGE (b:Node {id: 4}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 4}) MERGE (b:Node {id: 7}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 7}) MERGE (b:Node {id: 4}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 4}) MERGE (b:Node {id: 8}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 8}) MERGE (b:Node {id: 4}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 5}) MERGE (b:Node {id: 7}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 7}) MERGE (b:Node {id: 5}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 5}) MERGE (b:Node {id: 8}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 8}) MERGE (b:Node {id: 5}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 6}) MERGE (b:Node {id: 7}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 7}) MERGE (b:Node {id: 6}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 6}) MERGE (b:Node {id: 8}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 8}) MERGE (b:Node {id: 6}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 8}) MERGE (b:Node {id: 10}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 10}) MERGE (b:Node {id: 8}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 9}) MERGE (b:Node {id: 10}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 10}) MERGE (b:Node {id: 9}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 9}) MERGE (b:Node {id: 12}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 12}) MERGE (b:Node {id: 9}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 9}) MERGE (b:Node {id: 13}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 13}) MERGE (b:Node {id: 9}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 9}) MERGE (b:Node {id: 14}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 14}) MERGE (b:Node {id: 9}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 10}) MERGE (b:Node {id: 11}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 11}) MERGE (b:Node {id: 10}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 10}) MERGE (b:Node {id: 13}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 13}) MERGE (b:Node {id: 10}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 10}) MERGE (b:Node {id: 14}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 14}) MERGE (b:Node {id: 10}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 11}) MERGE (b:Node {id: 12}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 12}) MERGE (b:Node {id: 11}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 11}) MERGE (b:Node {id: 13}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 13}) MERGE (b:Node {id: 11}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 11}) MERGE (b:Node {id: 14}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 14}) MERGE (b:Node {id: 11}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 12}) MERGE (b:Node {id: 14}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 14}) MERGE (b:Node {id: 12}) CREATE (a)-[:RELATION]->(b);
    - |- 
        MATCH (a:Node {id: 9})-[r:RELATION]->(b:Node {id: 12}) DELETE r;
        MATCH (a:Node {id: 12})-[r:RELATION]->(b:Node {id: 9}) DELETE r;
        MATCH (a:Node {id: 9})-[r:RELATION]->(b:Node {id: 14}) DELETE r;
        MATCH (a:Node {id: 14})-[r:RELATION]->(b:Node {id: 9}) DELETE r;
        MATCH (a:Node {id: 10})-[r:RELATION]->(b:Node {id: 13}) DELETE r;
        MATCH (a:Node {id: 13})-[r:RELATION]->(b:Node {id: 10}) DELETE r;
        MERGE (a:Node {id: 0}) MERGE (b:Node {id: 13}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 13}) MERGE (b:Node {id: 0}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 3}) MERGE (b:Node {id: 9}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 9}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 10}) MERGE (b:Node {id: 12}) CREATE (a)-[:RELATION]->(b);
        MERGE (a:Node {id: 12}) MERGE (b:Node {id: 10}) CREATE (a)-[:RELATION]->(b);

cleanup: |-
    MATCH (n) DETACH DELETE n;
