setup: |-
    CREATE (n:Person {name:'Anna'}), (m:Person {name:'John'}), (k:Person {name:'Kim'}) CREATE (n)-[:IS_FRIENDS_WITH]->(m), (n)-[:IS_FRIENDS_WITH]->(k), (m)-[:IS_MARRIED_TO]->(k) CREATE (:User {cars: ['car1', 'car2', 'car3']}) CREATE (:List {listKey: [{inner: 'Map1'}, {inner: 'Map2'}]});
queries:
    - |-
        CALL export_util.json("/var/lib/memgraph/output.json");
    - |-
        MATCH (n) DETACH DELETE n;
    - |-
        CALL import_util.json("/var/lib/memgraph/output.json");
    - |-
        MATCH (n) RETURN count(n);
    - |-
        MATCH (n) RETURN count(n);
    - |-
        MATCH (n) RETURN count(n);


