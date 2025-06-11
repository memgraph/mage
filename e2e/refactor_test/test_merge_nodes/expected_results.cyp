// Test 1: Basic merge with combine strategy
// Expected: Combined properties from both nodes
// node.name | node.age | node.city | node.country
// "Alice"   | 30      | "New York"| "USA"

// Test 2: Merge with override strategy
// Expected: Properties from second node override first node
// node.name | node.age | node.city | node.country
// "Bob"     | 25      | null      | "USA"

// Test 3: Merge with discard strategy
// Expected: Only properties from first node
// node.name | node.age | node.city | node.country
// "Alice"   | 30      | "New York"| null

// Test 4: Merge with overwrite strategy
// Expected: Same as override strategy
// node.name | node.age | node.city | node.country
// "Bob"     | 25      | null      | "USA"

// Test 5: Merge nodes with different labels
// Expected: Combined properties and labels
// node.name | node.salary | node.department | labels(node)
// "David"   | 50000      | "IT"           | ["Employee", "Manager"]

// Test 6: Merge nodes with relationships
// Expected: Merged node with all relationships
// node.name | out_degree | in_degree
// "Alice"   | 2         | 2

// Test 7: Error case - empty list
// Expected: Error message about empty list

// Test 8: Error case - invalid property strategy
// Expected: Error message about invalid property strategy

// Test 9: Case insensitive property strategy
// Expected: Same as Test 1 (combine strategy)
// node.name | node.age | node.city | node.country
// "Alice"   | 30      | "New York"| "USA"

// Test 10: Merge with relationship strategy
// Expected: Merged node with all relationships
// node.name | out_degree | in_degree
// "Alice"   | 2         | 2 