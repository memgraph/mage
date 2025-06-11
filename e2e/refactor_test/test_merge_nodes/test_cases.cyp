// Test 1: Basic merge with combine strategy
MATCH (n1:Person {name: 'Alice'}), (n2:Person {name: 'Bob'})
CALL refactor.merge_nodes([n1, n2], {properties: 'combine'}) YIELD node
RETURN node.name, node.age, node.city, node.country;

// Test 2: Merge with override strategy
MATCH (n1:Person {name: 'Alice'}), (n2:Person {name: 'Bob'})
CALL refactor.merge_nodes([n1, n2], {properties: 'override'}) YIELD node
RETURN node.name, node.age, node.city, node.country;

// Test 3: Merge with discard strategy
MATCH (n1:Person {name: 'Alice'}), (n2:Person {name: 'Bob'})
CALL refactor.merge_nodes([n1, n2], {properties: 'discard'}) YIELD node
RETURN node.name, node.age, node.city, node.country;

// Test 4: Merge with overwrite strategy (alias for override)
MATCH (n1:Person {name: 'Alice'}), (n2:Person {name: 'Bob'})
CALL refactor.merge_nodes([n1, n2], {properties: 'overwrite'}) YIELD node
RETURN node.name, node.age, node.city, node.country;

// Test 5: Merge nodes with different labels
MATCH (n4:Employee {name: 'David'}), (n5:Manager {name: 'Eve'})
CALL refactor.merge_nodes([n4, n5], {properties: 'combine'}) YIELD node
RETURN node.name, node.salary, node.department, labels(node);

// Test 6: Merge nodes with relationships
MATCH (n1:Person {name: 'Alice'}), (n2:Person {name: 'Bob'}), (n3:Person {name: 'Charlie'})
CALL refactor.merge_nodes([n1, n2, n3], {properties: 'combine'}) YIELD node
RETURN node.name, size((node)-[]->()) as out_degree, size([]->(node)) as in_degree;

// Test 7: Error case - empty list
CALL refactor.merge_nodes([], {properties: 'combine'}) YIELD node
RETURN node;

// Test 8: Error case - invalid property strategy
MATCH (n1:Person {name: 'Alice'}), (n2:Person {name: 'Bob'})
CALL refactor.merge_nodes([n1, n2], {properties: 'invalid'}) YIELD node
RETURN node;

// Test 9: Case insensitive property strategy
MATCH (n1:Person {name: 'Alice'}), (n2:Person {name: 'Bob'})
CALL refactor.merge_nodes([n1, n2], {properties: 'COMBINE'}) YIELD node
RETURN node.name, node.age, node.city, node.country;

// Test 10: Merge with relationship strategy
MATCH (n1:Person {name: 'Alice'}), (n2:Person {name: 'Bob'})
CALL refactor.merge_nodes([n1, n2], {properties: 'combine', relationships: 'merge'}) YIELD node
RETURN node.name, size((node)-[]->()) as out_degree, size([]->(node)) as in_degree; 