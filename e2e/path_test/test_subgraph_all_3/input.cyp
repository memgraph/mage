CREATE (A:Node {name: 'A'}) CREATE (B:Node {name: 'B'}) CREATE (C:Node {name: 'C'}) CREATE (D:Node {name: 'D'}) CREATE (E:Node {name: 'E'}) CREATE (A)-[:CONNECTED_TO]->(B) CREATE (A)-[:CONNECTED_TO]->(C) CREATE (C)-[:CONNECTED_TO]->(D) CREATE (B)-[:CONNECTED_TO]->(E) CREATE (E)-[:CONNECTED_TO]->(D);
