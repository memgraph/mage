setup: |-
    CALL tgn.set_params("supervised", 5, 2, "graph_attn", 100, 100, 7, 10, 100, 5, "identity", "last", "gru", 1);
    CREATE TRIGGER create_embeddings ON --> CREATE BEFORE COMMIT EXECUTE CALL tgn.update(createdEdges) YIELD *;

queries:
    - |-
        MERGE (a:User {id: 2, features:[0.99,0.96,0.34,0.38,0.20,0.37,0.14,0.01,0.03,0.32]}) MERGE (b:Item {id: 2, features:[0.99,0.96,0.34,0.38,0.20,0.37,0.14,0.01,0.03,0.32]}) CREATE (a)-[:CLICKED {features:[-0.18,-0.18,-0.94,-0.38,0.00,-0.64,1.05]}]->(b);
        MERGE (a:User {id: 2, features:[0.99,0.96,0.34,0.38,0.20,0.37,0.14,0.01,0.03,0.32]}) MERGE (b:Item {id: 2, features:[0.99,0.96,0.34,0.38,0.20,0.37,0.14,0.01,0.03,0.32]}) CREATE (a)-[:CLICKED {features:[-0.18,-0.18,-0.94,-0.38,0.00,-0.64,1.05]}]->(b);
        MERGE (a:User {id: 3, features:[0.09,0.23,0.83,0.06,0.66,0.84,0.26,0.73,0.57,0.80]}) MERGE (b:Item {id: 3, features:[0.09,0.23,0.83,0.06,0.66,0.84,0.26,0.73,0.57,0.80]}) CREATE (a)-[:CLICKED {features:[-0.18,-0.18,-0.94,-0.38,0.00,-0.64,1.05]}]->(b);
        MERGE (a:User {id: 2, features:[0.99,0.96,0.34,0.38,0.20,0.37,0.14,0.01,0.03,0.32]}) MERGE (b:Item {id: 2, features:[0.99,0.96,0.34,0.38,0.20,0.37,0.14,0.01,0.03,0.32]}) CREATE (a)-[:CLICKED {features:[-0.18,-0.18,-0.94,-0.38,0.00,-0.64,1.05]}]->(b);
    - |-
       MERGE (a:User {id: 3, features:[0.09,0.23,0.83,0.06,0.66,0.84,0.26,0.73,0.57,0.80]}) MERGE (b:Item {id: 3, features:[0.09,0.23,0.83,0.06,0.66,0.84,0.26,0.73,0.57,0.80]}) CREATE (a)-[:CLICKED {features:[-0.18,-0.18,-0.94,-0.38,0.00,-0.64,1.05]}]->(b);
cleanup: |-
    CALL tgn.reset() YIELD *;
    DROP TRIGGER create_embeddings;
