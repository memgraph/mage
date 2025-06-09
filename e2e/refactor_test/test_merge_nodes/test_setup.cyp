// Create test nodes with different properties and labels
CREATE (n1:Person {name: 'Alice', age: 30, city: 'New York'})
CREATE (n2:Person {name: 'Bob', age: 25, country: 'USA'})
CREATE (n3:Person {name: 'Charlie', age: 35, city: 'London'})

// Create relationships between nodes
CREATE (n1)-[:KNOWS {since: 2020}]->(n2)
CREATE (n2)-[:WORKS_WITH {project: 'Project X'}]->(n3)
CREATE (n3)-[:FRIENDS_WITH {since: 2019}]->(n1)

// Create additional nodes for testing different scenarios
CREATE (n4:Employee {name: 'David', salary: 50000})
CREATE (n5:Manager {name: 'Eve', department: 'IT'})
CREATE (n6:Person {name: 'Frank', age: 40})

// Create relationships for additional nodes
CREATE (n4)-[:REPORTS_TO]->(n5)
CREATE (n5)-[:MANAGES]->(n6)
CREATE (n6)-[:COLLEAGUE]->(n4); 