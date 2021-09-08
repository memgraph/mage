- |-
    CALL node2vec_online.set_streamwalk_updater(7200, 2, 0.9, 604800, 2, False) YIELD *;
    CALL node2vec_online.set_word2vec_learner(2,0.01,True,1) YIELD *;
- |-
    CALL node2vec_online.reset() YIELD *;
    CALL node2vec_online.set_streamwalk_updater(7200, 2, 0.9, 604800, 2, False) YIELD *;
    CALL node2vec_online.set_word2vec_learner(2,0.01,True,1) YIELD *;
    MERGE (a:Node {id: 0}) MERGE (b:Node {id: 1}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 1}) MERGE (b:Node {id: 2}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 2}) MERGE (b:Node {id: 0}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 4}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 5}) CREATE (a)-[:RELATION]->(b);
    MATCH (n)-[e]->(m)
    WITH COLLECT(e) as edges
    CALL node2vec_online.update(edges) YIELD *
    WITH 1 as x
    RETURN x;
    CALL node2vec_online.reset() YIELD *;
- |-
    CALL node2vec_online.reset() YIELD *;
    CALL node2vec_online.set_streamwalk_updater(7200, 2, 0.9, 604800, 2, False) YIELD *;
    CALL node2vec_online.set_word2vec_learner(2,0.01,True,1) YIELD *;
    MERGE (a:Node {id: 0}) MERGE (b:Node {id: 1}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 1}) MERGE (b:Node {id: 2}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 2}) MERGE (b:Node {id: 0}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 4}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 5}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 4}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
    MATCH (n)-[e]->(m)
    WITH COLLECT(e) as edges
    CALL node2vec_online.update(edges) YIELD *
    WITH 1 as x
    RETURN x;
- |-
    CALL node2vec_online.reset() YIELD *;
    CALL node2vec_online.set_streamwalk_updater(7200, 2, 0.9, 604800, 2, False) YIELD *;
    CALL node2vec_online.set_word2vec_learner(2,0.01,True,1) YIELD *;
    MERGE (a:Node {id: 0}) MERGE (b:Node {id: 1}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 1}) MERGE (b:Node {id: 2}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 2}) MERGE (b:Node {id: 0}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 4}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 3}) MERGE (b:Node {id: 5}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 4}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
    MERGE (a:Node {id: 6}) MERGE (b:Node {id: 3}) CREATE (a)-[:RELATION]->(b);
    MATCH (n)-[e]->(m)
    WITH COLLECT(e) as edges
    CALL node2vec_online.update(edges) YIELD *
    WITH 1 as x
    RETURN x;

