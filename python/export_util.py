import csv
import io
import json as js
import mgp

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from gqlalchemy import Memgraph
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


def get_graph(
    ctx: mgp.ProcCtx, write_properties: bool
) -> List[Union[Node, Relationship]]:
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = [label.name for label in vertex.labels]
        properties = (
            {
                key: convert_to_isoformat(vertex.properties.get(key))
                for key in vertex.properties.keys()
            }
            if write_properties
            else {}
        )

        nodes.append(Node(vertex.id, labels, properties).get_dict())

        for edge in vertex.out_edges:
            properties = (
                {
                    key: convert_to_isoformat(edge.properties.get(key))
                    for key in edge.properties.keys()
                }
                if write_properties
                else {}
            )

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


def get_graph_from_map(graph_map: map, write_properties: bool):
    graph = list()
    for node in graph_map.get("nodes"):
        graph.append(
            Node(node.id, node.labels, node.properties if write_properties else {})
        )
    for relationship in graph_map.get("relationships"):
        graph.append(
            Relationship(
                relationship.to_vertex,
                relationship.id,
                relationship.type,
                relationship.properties if write_properties else {},
                relationship.from_vertex,
            )
        )
    return graph


def json_dump_to_file(graph: List[Union[Node, Relationship]], path: str):
    try:
        with open(path, "w") as outfile:
            js.dump(graph, outfile, indent=Parameter.STANDARD_INDENT.value, default=str)
    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
        )
    except Exception:
        raise OSError("Could not open or write to the file.")


@mgp.read_proc
def json(
    ctx: mgp.ProcCtx, path: str = "", config: mgp.Map = {}
) -> mgp.Record(path=str, data=str):
    """
    Procedure to export the whole database to a JSON file.

    Parameters
    ----------
    context : mgp.ProcCtx
        Reference to the context execution.
    path : str
        Path to the JSON file containing the exported graph database.
    config : mgp.Map
        stream (bool) = False: Flag to export the graph data to a stream.
        write_properties (bool) = False: Flag to keep node and relationship properties. By default set to true.
    """

    graph = get_graph(ctx, config.get("write_properties"))
    if path:
        json_dump_to_file(graph, path)

    return mgp.Record(
        path=path,
        data=js.dumps(graph) if config.get("stream") else "",
    )


@mgp.read_proc
def json_graph(
    ctx: mgp.ProcCtx, graph: mgp.Map, path: str = "", config: mgp.Map = {}
) -> mgp.Record(path=str, data=str):
    """
    Procedure to export the given graph to a JSON file. The graph is given with a map that contains keys "nodes" and "relationships".

    Parameters
    ----------
    graph : Map
        A map that contains a list of nodes at the key "nodes" and a list of relationships at the key "relationships"
    path : str
        Path to the JSON file containing the exported graph database.
    config : mgp.Map
        stream (bool) = False: Flag to export the graph data to a stream.
        write_properties (bool) = False: Flag to keep node and relationship properties. By default set to true.
    """
    graph = get_graph_from_map(graph, config.get("write_properties"))
    if path:
        json_dump_to_file(graph, path)

    return mgp.Record(
        path=path,
        data=js.dumps(graph) if config.get("stream") else "",
    )


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


@mgp.read_proc
def csv_query(
    context: mgp.ProcCtx,
    query: str,
    file_path: str = "",
    stream: bool = False,
) -> mgp.Record(file_path=str, data=str):
    """
    Procedure to export query results to a CSV file.
    Args:
        context (mgp.ProcCtx): Reference to the context execution.
        query (str): A query from which the results will be saved to a CSV file.
        file_path (str, optional): A path to the CSV file where the query results will be exported. Defaults to an empty string.
        stream (bool, optional): A value which determines whether a stream of query results in a CSV format will be returned.
    Returns:
        mgp.Record(
            file_path (str): A path to the CSV file where the query results are exported. If file_path is not provided, the output will be an empty string.
            data (str): A stream of query results in a CSV format.
        )
    Raises:
        Exception: If neither file nor config are provided, or if only config is provided with stream set to False. Also if query yields no results or if the database is empty.
        PermissionError: If you provided file path that you have no permissions to write at.
        csv.Error: If an error occurred while writing into stream or CSV file.
        OSError: If the file can't be opened or written to.
    """

    # file or config have to be provided
    if not file_path and not stream:
        raise Exception("Please provide file name and/or config.")

    # only config provided with stream set to false
    if not file_path and not stream:
        raise Exception(
            "If you provided only stream value, it has to be set to True to get any results."
        )

    memgraph = Memgraph()
    results = list(memgraph.execute_and_fetch(query))

    # if query yields no result
    if not len(results):
        raise Exception(
            "Your query yields no results. Check if the database is empty or rewrite the provided query."
        )

    result_keys = list(results[0])
    data_list = [result_keys] + [list(result.values()) for result in results]
    data = ""

    if file_path:
        save_file(file_path, data_list)

    if stream:
        data = csv_to_stream(data_list)

    return mgp.Record(file_path=file_path, data=data)
