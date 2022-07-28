from gqlalchemy import Memgraph
import numpy as np




memgraph = Memgraph("127.0.0.1", 7687)


def drop_some_nodes():
    results = memgraph.execute_and_fetch(
        """
        MATCH (n) RETURN n.id AS ids ;
        """
    )


    total = sum(1 for _ in results)
    mask = np.zeros((total))

    i = 0
    while i < total * 0.1:
        mask[i] = 1
        i+=1

    mask = np.random.permutation(mask)


    for i in range(total):
        if mask[i]:
            memgraph.execute(
                f"MATCH (n {{id: {i}}}) DETACH DELETE n ;"
            )
 

def drop_specific_node_edges(id: int):
    results = memgraph.execute(
                f"MATCH (n {{id: {id}}}) DETACH DELETE n ;"
    )

    
