from database import Memgraph, Node
from typing import Dict, List


def create_node(node_id, properties):
    node = Node(node_id=node_id, properties=properties)
    node.properties['id'] = node_id

    return node


def insert_nodes(
        connection: Memgraph,
        nodes: Node
):
    for node in nodes:
        query = f"""CREATE (a {str_props(node)});"""
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


def str_props(
        node: Node
):
    stringified_props = ", ".join([f"{key}: {node.properties[key]}" for key in node.properties.keys()])
    return f"{{{stringified_props}}}"


def str_nodes(nodes, group_name):
    nodes_to_match = ", ".join([f"({group_name}{idx} {str_props(node)})" for idx, node in enumerate(nodes)])
    return nodes_to_match


def tsp(
        connection: Memgraph,
        solving_method: str = None
) -> (List, List):
    """
    Returns the result of 'distance_calculator.single()' module from Memgraph.

    The result is written in following format:

    result = distance

    :param connection: Connection to Memgraph
    :param solving_method: solving method from the possible ones
    :return: sources - starting points
    :return: destinations - ending point
    """

    if solving_method is None:
        call = f"""CALL tsp.solve(points)"""
    else:
        call = f"""CALL tsp.solve(points, '{solving_method}')"""

    query = f"""
    MATCH (n)
    WITH COLLECT(n) AS points
    {call}
    YIELD sources, destinations
    RETURN sources, destinations;
    """
    result = connection.execute_and_fetch(query)

    sources = None
    destinations = None
    for r in result:
        sources = r['sources']
        destinations = r['destinations']

    return sources, destinations
