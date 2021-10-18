
match (n)-[e]->(m)
with collect(e) as edges
call node2vec.get_embeddings(edges) YIELD * RETURN *;