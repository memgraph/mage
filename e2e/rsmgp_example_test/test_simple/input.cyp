MATCH (n) DETACH DELETE n;
CREATE ({age: 29}), (:L1 {age: 31, name: "John"}), (:L2:L3 {name: "Phil"}), (:L4:L5:L6);
