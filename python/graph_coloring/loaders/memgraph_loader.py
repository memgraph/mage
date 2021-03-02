from telco import query
from telco.graph import Graph
from collections import defaultdict


def load_memgraph() -> Graph:
    nodes = []
    nodes_command = 'MATCH (a: Cell) RETURN a;'
    for d in list(query(nodes_command)):
        nodes.append(d['a'].properties['id'])

    adj_list = defaultdict(list)
    edges_command = 'MATCH (a: Cell)-[e: CLOSE_TO]->(b: Cell) RETURN a,e,b;'
    for d in list(query(edges_command)):
        w = d['e'].properties['weight']
        a = d['a'].properties['id']
        b = d['b'].properties['id']
        adj_list[a].append((b, w))
        adj_list[b].append((a, w))

    return Graph(nodes, adj_list)
