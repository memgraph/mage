from telco import query
from telco.graph import Graph
from telco.components.individual import Individual


def write_to_memgraph(graph: Graph, indv: Individual) -> None:
    for n in graph.nodes:
        node = graph.label(n)
        c = 'MATCH (a: Cell {{id: {}}}) SET a.PCI = {} RETURN a'.format(node, indv[n])
        query(c)

    for i in graph.nodes:
        for j in graph[i]:
            node_1 = graph.label(i)
            node_2 = graph.label(j)
            if indv[i] == indv[j]:
                c = 'MATCH (a: Cell {{id: {}}})-[e: CLOSE_TO]->(b: Cell {{id: {}}}) '\
                    'SET e.conflict = TRUE'.format(node_1, node_2)
            else:
                c = 'MATCH (a: Cell {{id: {}}})-[e: CLOSE_TO]->(b: Cell {{id: {}}}) '\
                    'SET e.conflict = FALSE'.format(node_1, node_2)
            query(c)
