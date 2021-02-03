from database import Memgraph, Node
from typing import Dict, List


def create_node(node_id, properties, group=None):
    node = Node(node_id=node_id, properties=properties)
    node.properties['id'] = node_id

    if group is not None:
        node.properties['group'] = f"'{group}'"

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


def calculate_distance(
        connection: Memgraph,
        n1: Node,
        n2: Node,
        metrics: str = None
) -> float:
    """
    Returns the result of 'distance_calculator.single()' module from Memgraph.

    The result is written in following format:

    result = distance

    :param connection: Connection to Memgraph
    :param n1 first node
    :param n2 second node
    :param metrics tells which units to display the distance
    :return: Distance float result
    """

    if metrics is None:
        call = f"""CALL distance_calculator.single(n1, n2)"""
    else:
        call = f"""CALL distance_calculator.single(n1, n2, '{metrics}')"""

    query = f"""
    MATCH (n1 {str_props(n1)}), (n2 {str_props(n2)})
    {call}
    YIELD distance
    RETURN distance;
    """
    result = connection.execute_and_fetch(query)

    return list(result)[0]['distance']


def calculate_distance_multiple(
        connection: Memgraph,
        begin_nodes: List[Node],
        end_nodes: List[Node],
        metrics: str = None
) -> List:
    """
    Returns the result of 'distance_calculator.single()' module from Memgraph.

    The result is written in following format:

    result = distance

    :param connection: Connection to Memgraph
    :param begin_nodes starting nodes
    :param end_nodes ending nodes
    :param metrics tells which units to display the distance
    :return: Distances list of floats
    """

    if metrics is None:
        call = f"""CALL distance_calculator.multiple(begin_nodes, end_nodes)"""
    else:
        call = f"""CALL distance_calculator.multiple(begin_nodes, end_nodes, '{metrics}')"""

    query = f"""
    MATCH (n {{group:'a'}})
    WITH n
    ORDER BY n.id
    WITH COLLECT(n) AS begin_nodes
    MATCH (m {{group: 'b'}})
    WITH begin_nodes, m
    ORDER BY m.id
    WITH begin_nodes, COLLECT(m) AS end_nodes
    {call}
    YIELD distances
    RETURN distances;
    """
    result = connection.execute_and_fetch(query)

    distances = None
    for r in result:
        distances = r['distances']

    return distances