MATCH (n) DETACH DELETE n;
CREATE ({age: 29}), (:L1 {age: 31}), (:L2:L3 {age: 10}), (:L4:L5:L6 {age: 20});
