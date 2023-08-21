import csv
import io
import json as js
import mgp
import ast

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta

from typing import Any, Dict, List, Union

from mage.export_import_util.parameters import Parameter


@dataclass
class Node:
    id: int
    labels: list
    properties: dict

    def get_dict(self) -> dict:
        return {
            Parameter.ID.value: self.id,
            Parameter.LABELS.value: self.labels,
            Parameter.PROPERTIES.value: self.properties,
            Parameter.TYPE.value: Parameter.NODE.value,
        }


@dataclass
class Relationship:
    end: int
    id: int
    label: str
    properties: dict
    start: int

    def get_dict(self) -> dict:
        return {
            Parameter.END.value: self.end,
            Parameter.ID.value: self.id,
            Parameter.LABEL.value: self.label,
            Parameter.PROPERTIES.value: self.properties,
            Parameter.START.value: self.start,
            Parameter.TYPE.value: Parameter.RELATIONSHIP.value,
        }


def convert_to_isoformat(
    property: Union[
        None,
        str,
        bool,
        int,
        float,
        List[Any],
        Dict[str, Any],
        timedelta,
        time,
        datetime,
        date,
    ]
):
    if isinstance(property, timedelta):
        return Parameter.DURATION.value + str(property) + ")"

    elif isinstance(property, time):
        return Parameter.LOCALTIME.value + property.isoformat() + ")"

    elif isinstance(property, datetime):
        return Parameter.LOCALDATETIME.value + property.isoformat() + ")"

    elif isinstance(property, date):
        return Parameter.DATE.value + property.isoformat() + ")"

    else:
        return property


def get_graph(ctx: mgp.ProcCtx) -> List[Union[Node, Relationship]]:
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = [label.name for label in vertex.labels]
        properties = {
            key: convert_to_isoformat(vertex.properties.get(key))
            for key in vertex.properties.keys()
        }

        nodes.append(Node(vertex.id, labels, properties).get_dict())

        for edge in vertex.out_edges:
            properties = {
                key: convert_to_isoformat(edge.properties.get(key))
                for key in edge.properties.keys()
            }

            relationships.append(
                Relationship(
                    edge.to_vertex.id,
                    edge.id,
                    edge.type.name,
                    properties,
                    edge.from_vertex.id,
                ).get_dict()
            )

    return nodes + relationships


def get_graph_from_lists(node_list: list, relationship_list: list, write_properties: bool):

    graph = list()
    node_properties = list()
    node_prop_set = set()
    relationship_properties = list()
    relationship_prop_set = set()


    for node in node_list:
        for prop in node.properties:
            if not prop in node_prop_set:
                node_properties.append(prop)
                node_prop_set.add(prop)
        node_properties.sort()
        graph.append(

            Node(node.id, node.labels, node.properties if write_properties else {})

        )

    for relationship in relationship_list:
        for prop in relationship.properties:
            if not prop in relationship_prop_set:
                relationship_properties.append(prop)
                relationship_prop_set.add(prop)
        relationship_properties.sort()

        graph.append(

            Relationship(

                relationship.to_vertex.id,

                relationship.id,

                relationship.type.name,

                relationship.properties if write_properties else {},

                relationship.from_vertex.id,

            )

        )

    return graph, node_properties, relationship_properties

def csv_header(node_properties: list, relationship_properties: list) -> list:
    """
    This function creates the header for csv file 
    """
    header = ["_id","_labels"]

    for prop in node_properties:
        header.append(prop)

    header.extend(["_start", "_end", "_type"])

    for prop in relationship_properties:
        header.append(prop)

    return header

def csv_data_list(graph: list, node_properties: list, relationship_properties: list) -> list:

    """
    Function that parses graph into a data_list appropriate for csv writing
    """
    data_list = []
    header = csv_header(node_properties, relationship_properties)
    data_list.append(header)
    for element in graph:
        write_list = []
        IsNode = isinstance(element, Node)
        
        #id
        if(IsNode):
            write_list.extend([element.id, "".join(":" + label.name for label in element.labels)])
        else:
            write_list.extend(["", ""])

        #node_properties
        for prop in node_properties:
            if prop in element.properties and IsNode:
                if isinstance(element.properties[prop], (set, list, tuple, map)):
                    write_list.append(js.dumps(element.properties[prop]))
                else:   
                    write_list.append(element.properties[prop])
            else:
                write_list.append("")
        #relationship
        if(IsNode):
            #start, end, type
            write_list.extend(["", "", ""])
        else:
            #start, end, type
            write_list.extend([element.start, element.end, element.label])

        #relationship properties
        for prop in relationship_properties:
            if prop in element.properties and not IsNode:
                if isinstance(element.properties[prop], (set, list, tuple, map)):
                    write_list.append(js.dumps(element.properties[prop]))
                else:   
                    write_list.append(element.properties[prop])
            else:
                write_list.append("")
        
        data_list.append(write_list)

    return data_list


@mgp.read_proc
def csv_graph(
    nodes_list: mgp.List[mgp.Vertex], relationships_list: mgp.List[mgp.Edge], path: str = "", config: mgp.Map = {}

) -> mgp.Record(path=str, data=str):

    """
    Procedure to export the given graph to a csv file. The graph is given with two lists, one for nodes, and one for relationships.


    Parameters

    ----------

    nodes_list : List

        A list containing nodes of the graph
    
    relationships_list : List

        A list containing relationships of the graph

    path : str

        Path to the JSON file containing the exported graph database.

    config : mgp.Map

        stream (bool) = False: Flag to export the graph data to a stream.

        write_properties (bool) = False: Flag to keep node and relationship properties. By default set to true.

    """

    graph, node_properties, relationship_properties = get_graph_from_lists(nodes_list, relationships_list, True)
    data_list = csv_data_list(graph, node_properties, relationship_properties)
    f = open("/home/matija/Documents/file.csv","w")
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerows(data_list)    
    f.close()
    data = csv_to_stream(data_list)

    return mgp.Record(
        path=path,
        data=data,
    )


@mgp.read_proc
def json(ctx: mgp.ProcCtx, path: str) -> mgp.Record():
    """
    Procedure to export the whole database to a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file containing the exported graph database.
    """

    graph = get_graph(ctx)
    try:
        with open(path, "w") as outfile:
            js.dump(graph, outfile, indent=Parameter.STANDARD_INDENT.value, default=str)
    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
        )
    except Exception:
        raise OSError("Could not open or write to the file.")

    return mgp.Record()


@mgp.read_proc
def json_stream(ctx: mgp.ProcCtx) -> mgp.Record(stream=str):
    """
    Procedure to export the whole database to a stream.
    """
    return mgp.Record(stream=js.dumps(get_graph(ctx)))


def save_file(file_path: str, data_list: list):
    try:
        with open(
            file_path,
            "w",
            newline="",
            encoding="utf8",
        ) as f:
            writer = csv.writer(f)
            writer.writerows(data_list)
    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
        )
    except csv.Error as e:
        raise csv.Error(
            "Could not write to the file {}, stopped at line {}: {}".format(
                file_path, writer.line_num, e
            )
        )
    except Exception:
        raise OSError("Could not open or write to the file.")


def csv_to_stream(data_list: list) -> str:
    output = io.StringIO()
    try:
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(data_list)
    except csv.Error as e:
        raise csv.Error(
            "Could not write a stream, stopped at line {}: {}".format(
                writer.line_num, e
            )
        )
    return output.getvalue()


