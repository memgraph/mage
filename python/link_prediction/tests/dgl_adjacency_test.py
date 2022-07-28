import dgl
import numpy as np

def test_adjacency_matrix(graph: dgl.graph, adj_matrix: np.matrix) -> bool:
    """Tests whether the adjacency matrix correctly encodes edges from dgl graph

    Args:
        graph (dgl.graph): A reference to the original graph we are working with. 
        adj_matrix (np.matrix): Graph's adjacency matrix

    Returns:
        bool: True if adjacency matrix and graph are equivalent, False otherwise
    """


    if adj_matrix.shape[0] != graph.number_of_nodes() or \
        adj_matrix.shape[1] != graph.number_of_nodes():
        return False

    # To check that indeed adjacency matrix is equivalent to graph we need to check both directions to get bijection.

    # First test direction: graph->adj_matrix
    u, v = graph.edges()
    num_edges = graph.number_of_edges()

    for i in range(num_edges):
        v1, v2 =  u[i].item(), v[i].item()
        if adj_matrix[v1][v2] != 1.0:
            return False

    
    # Now test the direction adj_matrix->graph
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
             if adj_matrix[i][j] == 1.0:
                if graph.has_edges_between(i, j) is False:
                    return False

    # If we are here
    return True