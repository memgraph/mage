setup: |-
    CREATE (n:Person {name:'Anna'}), (m:Person {name:'John'}), (k:Person {name:'Kim'}) CREATE (n)-[:IS_FRIENDS_WITH]->(m), (n)-[:IS_FRIENDS_WITH]->(k), (m)-[:IS_MARRIED_TO]->(k);

queries:
    - |-
        CALL export_util.json("/var/lib/memgraph/output.json");
    - |-
        MATCH (n) DETACH DELETE n;
    - |-
        CALL import_util.json("/var/lib/memgraph/output.json");



