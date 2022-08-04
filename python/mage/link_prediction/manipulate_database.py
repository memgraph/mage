from multiprocessing.connection import wait
from operator import mod
from platform import node
import time
from gqlalchemy import Memgraph, Create, Node, Relationship
import numpy as np
import ast

from py import test


class MemgraphPrintDict:
    """Class that wraps dictionary by printing repr of value and key without quotes to be Cypher compatible."""

    def __init__(self, my_dict) -> None:
        self.my_dict = my_dict

    def __str__(self) -> str:
        s = ""
        items_len = len(self.my_dict.keys())
        for i, (key, value) in enumerate(self.my_dict.items()):
            s += key + ": " + repr(value)
            if i != items_len - 1:
                s += ", "
        return s


def labels_list_to_cypher(labels: list) -> str:
    """Generates Cypher labels list representation command.

    Args:
        labels (list): List of labels

    Returns:
        str: Cypher list representation
    """
    command = ""
    for label in labels:
        command += f":{label}"

    return command


def delete_edges(
    num_delete_edges: int,
    file_name: str,
    node_id_property: str,
    file_prediction_commands: str,
    file_test_command: str
) -> None:
    """Deletes num_delete_edges random relationships.
    Creates CREATE commands and saves it to the file so later relationships can be recreated.
    Creates PREDICT commands for each edge separately.
    CREATES PREDICT command for testing all edges at once. 
    Every time, old files are deleted(file_prediction_command, file_name and file_test_command)

    Args:
        num_delete_edges (int): Number of edges to be deleted.
        file_name (str): Name of the file where it should be saved.
        node_id_property (str): Property name where the id is saved.
        file_prediction_commands (str): File where prediction methods will be saved.
        file_test_command (str): Path to the file where test command should be saved.
    Returns:
        nothing
    """

    results = memgraph.execute_and_fetch(
        f"""
        MATCH (v1)-[e]->(v2) 
        RETURN v1, e, v2
        ORDER BY rand()
        LIMIT {num_delete_edges};
        """
    )

    create_commands, prediction_commands = [], []
    vertex_ids_delete = []  # from node_id_property

    vertex_ids_test_src, vertex_ids_test_dest = [], []  # source and destination vertices from memgraph's _id property

    for result in results:
        # Handle first node
        v1_id, v1_properties, v1_labels = result["v1"]._id, MemgraphPrintDict(result["v1"]._properties), labels_list_to_cypher(list(result["v1"]._labels))
        vertex_ids_delete.append(v1_properties.my_dict[node_id_property])
        vertex_ids_test_src.append(v1_id)

        # Handle second node
        v2_id, v2_properties, v2_labels = result["v2"]._id, MemgraphPrintDict(result["v2"]._properties), labels_list_to_cypher(list(result["v2"]._labels))
        vertex_ids_test_dest.append(v2_id)

        # del v2_properties.my_dict["features"]
        # Handle edge
        edge_properties, edge_labels = (MemgraphPrintDict(result["e"]._properties), result["e"]._type,)

        # Create command for later, and save it to some my file.
        create_command = f"""MATCH (v1{v1_labels} {{{node_id_property}: {v1_properties.my_dict[node_id_property]}}})
        MATCH (v2{v2_labels} {{{node_id_property}: {v2_properties.my_dict[node_id_property]}}}) 
        CREATE (v1)-[e:{edge_labels} {{{edge_properties}}}]->(v2)
        RETURN v1, e, v2;
        """

        # print(create_command)
        create_commands.append(create_command)

        # Delete command
        delete_command = f"""
            MATCH (v1{v1_labels} {{{node_id_property}: {v1_properties.my_dict[node_id_property]}}})-[e]->(v2{v2_labels} {{{node_id_property}: {v2_properties.my_dict[node_id_property]}}}) 
            DELETE e;
        """
        # print("DELETE COMMAND: ")
        # print(delete_command)

        memgraph.execute(delete_command)

        # Add support for prediction methods
        predict_command = f"""MATCH (v1{v1_labels} {{{node_id_property}: {v1_properties.my_dict[node_id_property]}}})
        MATCH (v2{v2_labels} {{{node_id_property}: {v2_properties.my_dict[node_id_property]}}}) 
        CALL link_prediction.predict(v1, v2)
        YIELD *
        RETURN *;
        """
        prediction_commands.append(predict_command)

    # Save create commands
    with open(file=file_name, mode="w") as f:
        for comm in create_commands:
            f.write("%s\n" % comm)

    # Save predict commands
    with open(file=file_prediction_commands, mode="w") as f:
        for pred_comm in prediction_commands:
            f.write("%s\n" % pred_comm)

    # Test command
    test_command = f"""CALL link_prediction.test({vertex_ids_test_src}, {vertex_ids_test_dest})
    YIELD *
    RETURN *;
    """

    print("Test command: ", test_command)
    with open(file=file_test_command, mode="w") as f:
        f.write(test_command)


    return results


def read_commands_script(file_name: str) -> None:
    """Reads commands from file name and executes them.

    Args:
        file_name (str): _description_
    """
    with open(file=file_name, mode="r+") as f:
        command = ""
        while True:
            line = f.readline()
            if not line:
                break
            elif line.isspace() is True:  # empty line so execute command
                print("Command: ")
                print(command)
                memgraph.execute(command)
                time.sleep(5)
                command = ""  # annulate old command
            else:
                command += line

        f.truncate(0)


if __name__ == "__main__":
    memgraph = Memgraph("127.0.0.1", 7687)
    file_name = (
        "./commands/cora_creation_commands.txt"  # where CREATE commands will be created
    )
    prediction_file_name = "./commands/prediction_commands.txt"  # where PREDICTION commands will be created
    test_file_name = "./commands/test_command.txt"
    delete_edges(
        num_delete_edges=1000,
        file_name=file_name,
        node_id_property="id",
        file_prediction_commands=prediction_file_name,
        file_test_command=test_file_name
    )
    # read_commands_script(file_name=file_name)
