from database import Memgraph, Node
from typing import Dict, List


def make_graph(
        connection: Memgraph,
        graph: Dict
):
    """
    Creates a graph from dictionary-like description. The only property for created graphs is called
    "id" which stores the name.

    Graph description can be written like:

    graph = {
        "node_1:Node": [node_2:Node:Edge]
    }
    :param connection: Connection to  Memgraph database
    :param graph: Graph description
    """
    for vertex_a, adj_a in graph.items():
        name_a, type_a = vertex_a.split(":")
        for vertex_b in adj_a:
            name_b, type_b, type_edge = vertex_b.split(":")
            query = f"""MERGE (a:{type_a} {{id: "{name_a}"}})
                        MERGE (b:{type_b} {{id: "{name_b}"}})
                        CREATE (a)-[e:{type_edge}]->(b);"""

            connection.execute_query(query)


def delete_all(
        connection: Memgraph
):
    """
    Deletes all nodes in Memgraph database.

    :param connection: Memgraph database connector
    """
    query = f"""MATCH (n) DETACH DELETE n;"""
    connection.execute_query(query)


def str_node(
        node: Node
):
    """
    Changing from Node Object to the graph description.

    name:Label

    :param node: Node Object
    """
    return f"{node.properties['id']}:{list(node.labels)[0]}"


def collapse(
        connection: Memgraph,
        vertex: str,
        edge_types: List[str],
        pseudo_labels: List[str] = None
) -> List[Dict]:
    """
    Returns the result of 'collapse.collapse()' module from Memgraph.

    The result is written in following format:

    result = [
        {"from_vertex": "node_1:Node", "path": ["node_2:Node", "node_3:Node"], "to_vertex": "node_4:Node"}
    ]

    :param connection: Connection to Memgraph
    :param vertex: String node label
    :param edge_types: List of edge types
    :param pseudo_labels: List of pseudo labels
    :return: List of dictionaries as a result
    """
    if not pseudo_labels:
        query = f"""CALL collapse.collapse(nodes, {str(edge_types)})"""
    else:
        query = f"""CALL collapse.collapse(nodes, {str(edge_types)}, {str(pseudo_labels)})"""

    query = f"""
    MATCH (n:{vertex}) WITH COLLECT(n) AS nodes
    {query}
    YIELD from_vertex, path, to_vertex
    RETURN from_vertex, nodes(path), to_vertex;
    """
    result = connection.execute_and_fetch(query)
    translated_results = []
    for r in result:
        translated_results.append(
            {
                'from_vertex': str_node(r['from_vertex']),
                'path': [str_node(node) for node in r['nodes(path)']],
                'to_vertex': str_node(r['to_vertex'])
            }
        )
    return translated_results


def groups(
        connection,
        vertex,
        edge_types,
        pseudo_labels=None
) -> List[Dict]:
    """
    Returns the result of 'collapse.groups()' module from Memgraph.

    The result is written in following format:

    result = [
        {"top_vertex": "node_1:Node", "collapsed_vertices": ["node_1:Node", " node_2:Node", "node_3:Node"]}
    ]

    :param connection: Connection to Memgraph
    :param vertex: String node label
    :param edge_types: List of edge types
    :param pseudo_labels: List of pseudo labels
    :return: List of dictionaries as a result
    """
    if not pseudo_labels:
        query = f"""CALL collapse.groups(nodes, {str(edge_types)})"""
    else:
        query = f"""CALL collapse.groups(nodes, {str(edge_types)}, {str(pseudo_labels)})"""

    query = f"""
    MATCH (n:{vertex}) WITH COLLECT(n) AS nodes
    {query}
    YIELD *
    RETURN top_vertex, collapsed_vertices;
    """
    result = connection.execute_and_fetch(query)
    translated_results = []
    for r in result:
        translated_results.append(
            {
                'top_vertex': str_node(r['top_vertex']),
                'collapsed_vertices': [str_node(node) for node in r['collapsed_vertices']],
            }
        )
    return translated_results
