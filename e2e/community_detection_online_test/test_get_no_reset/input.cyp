MERGE (a: Node {id: 0}) MERGE (b: Node {id: 1}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 0}) MERGE (b: Node {id: 2}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 1}) MERGE (b: Node {id: 2}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 2}) MERGE (b: Node {id: 3}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 3}) MERGE (b: Node {id: 4}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 3}) MERGE (b: Node {id: 5}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 4}) MERGE (b: Node {id: 5}) CREATE (a)-[r: Relation]->(b);
CALL community_detection_online.set() YIELD node, community_id RETURN node.id AS node_id, community_id ORDER BY node_id;
CALL community_detection_online.get() YIELD node, community_id RETURN node.id AS node_id, community_id ORDER BY node_id;
CREATE (n: Node {id: 15});
MERGE (a: Node {id: 15}) MERGE (b: Node {id: 8}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 15}) MERGE (b: Node {id: 10}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 15}) MERGE (b: Node {id: 14}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 2}) MERGE (b: Node {id: 4}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 2}) MERGE (b: Node {id: 6}) CREATE (a)-[r: Relation]->(b);
MERGE (a: Node {id: 1}) MERGE (b: Node {id: 7}) CREATE (a)-[r: Relation]->(b);
MATCH (n: Node {id: 5}) DETACH DELETE n;
