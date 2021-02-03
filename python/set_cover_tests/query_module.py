from database import Memgraph, Node, Relationship
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


def insert_relationship(
        connection: Memgraph,
        node1: Node,
        node2: Node
):
    query = f"""MERGE (a {str_props(node1)})
                MERGE (b {str_props(node2)})
                CREATE (a)-[:BELONGS_TO]->(b)"""
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


def set_cover(
        connection: Memgraph,
        solving_method: str = None
) -> (List, List):
    """
    Returns containing sets of all elements

    The result is written in following format:

    result = list of containing sets

    :param connection: Connection to Memgraph
    :param solving_method: solving method from the possible ones
    :return: List of contained set nodes
    """

    if solving_method == 'greedy':
        call = f"""CALL set_cover.greedy(elements, sets)"""
    else:
        call = f"""CALL set_cover.cp_solve(elements, sets)"""

    query = f"""
    MATCH (n)-[:BELONGS_TO]->(m)
    WITH COLLECT(n) AS elements, COLLECT(m) AS sets
    {call}
    YIELD resulting_sets
    RETURN resulting_sets;
    """
    result = connection.execute_and_fetch(query)

    resulting_sets = None
    for r in result:
        resulting_sets = r['resulting_sets']

    return resulting_sets
