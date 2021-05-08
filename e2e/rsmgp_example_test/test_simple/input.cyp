MATCH (n) DETACH DELETE n;
CREATE ({age: 30}), (:L1 {name: "name1"}), (:L2:L3 {name: "name2"}), (:L4:L5:L6 {name: "bla"})
