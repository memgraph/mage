CREATE (v1:PAPER {id: 10, features: [1, 2, 3]});
CREATE (v2:PAPER {id: 11, features: [1.54, 0.3, 1.78]});
CREATE (v3:BOOK {id: 12, features: [0.5, 1, 4.5]});
MATCH (v1:PAPER {id: 10}), (v2:PAPER {id: 11}) CREATE (v1)-[e:CITES {}]->(v2);
MATCH (v2:PAPER {id: 11}), (v3:BOOK {id: 12}) CREATE (v2)-[e:CITES {}]->(v3);