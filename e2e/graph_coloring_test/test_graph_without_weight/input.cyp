CREATE (:Cell {name: "Cell_0", id: 0});
CREATE (:Cell {name: "Cell_1", id: 1});
CREATE (:Cell {name: "Cell_2", id: 2});
CREATE (:Cell {name: "Cell_3", id: 3});
CREATE (:Cell {name: "Cell_4", id: 4});
MATCH (a:Cell {id: 0}), (b:Cell {id: 1}) CREATE (a)-[e:Edge]->(b);
MATCH (a:Cell {id: 0}), (b:Cell {id: 2}) CREATE (a)-[e:Edge]->(b);
MATCH (a:Cell {id: 1}), (b:Cell {id: 3}) CREATE (a)-[e:Edge]->(b);
MATCH (a:Cell {id: 1}), (b:Cell {id: 4}) CREATE (a)-[e:Edge]->(b);
MATCH (a:Cell {id: 3}), (b:Cell {id: 4}) CREATE (a)-[e:Edge]->(b);
