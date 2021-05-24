MATCH (n) DETACH DELETE n;
CREATE (v1 {age: 29}), (v2:L1 {age: 31.3}), (v3:L2:L3 {name: "Phil"}), (v4:L4:L5:L6), (v1)-[:E1]->(v2);
