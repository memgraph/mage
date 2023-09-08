import csv
import io
import json as js
import mgp
import gqlalchemy

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from gqlalchemy import Memgraph
from typing import Any, Dict, List, Union
from math import floor

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


def to_duration_iso_format(value: timedelta) -> str:
    """Converts timedelta to ISO-8601 duration: P<date>T<time>"""
    date_parts: List[str] = []
    time_parts: List[str] = []

    if value.days != 0:
        date_parts.append(f"{abs(value.days)}D")

    if value.seconds != 0 or value.microseconds != 0:
        abs_seconds = abs(value.seconds)
        hours = floor(abs_seconds / 3600)
        minutes = floor((abs_seconds - hours * 3600) / 60)
        seconds = abs_seconds - hours * 3600 - minutes * 60
        microseconds = value.microseconds

        if hours > 0:
            time_parts.append(f"{hours}H")
        if minutes > 0:
            time_parts.append(f"{minutes}M")
        if seconds > 0 or microseconds > 0:
            microseconds_part = (
                f".{abs(value.microseconds)}" if value.microseconds != 0 else ""
            )
            time_parts.append(f"{seconds}{microseconds_part}S")

    date_duration_str = "".join(date_parts)
    time_duration_str = f'T{"".join(time_parts)}' if time_parts else ""

    return f"P{date_duration_str}{time_duration_str}"


def convert_to_cypher_format(
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
) -> str:
    if isinstance(property, timedelta):
        return f"duration('{to_duration_iso_format(property)}')"

    elif isinstance(property, time):
        return f"localTime('{property.isoformat()}')"

    elif isinstance(property, datetime):
        return f"localDateTime('{property.isoformat()}')"

    elif isinstance(property, date):
        return f"date('{property.isoformat()}')"

    elif isinstance(property, str):
        return f"'{property}'"

    elif isinstance(property, tuple):  # list
        return (
            "[" + ", ".join([convert_to_cypher_format(item) for item in property]) + "]"
        )

    elif isinstance(property, dict):
        return (
            "{"
            + ", ".join(
                [f"{k}: {convert_to_cypher_format(v)}" for k, v in property.items()]
            )
            + "}"
        )

    return str(property)


def get_graph_for_cypher(
    ctx: mgp.ProcCtx, write_properties: bool
) -> List[Union[Node, Relationship]]:
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = [label.name for label in vertex.labels]
        properties = (
            {
                key: convert_to_cypher_format(vertex.properties.get(key))
                for key in vertex.properties.keys()
            }
            if write_properties
            else {}
        )

        nodes.append(Node(vertex.id, labels, properties))

        for edge in vertex.out_edges:
            properties = (
                {
                    key: convert_to_cypher_format(edge.properties.get(key))
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
                )
            )

    return nodes + relationships


@mgp.read_proc
def cypher_all(
    ctx: mgp.ProcCtx,
    path: str = "",
    config: mgp.Map = {},
) -> mgp.Record(path=str, data=str):
    """Exports the graph in cypher with all the constraints, indexes and triggers.
    Args:
        context (mgp.ProcCtx): Reference to the context execution.
        path (str): A path to the file where the query results will be exported. Defaults to an empty string.
        config : mgp.Map
            stream (bool) = False: Flag to export the graph data to a stream.
            write_properties (bool) = True: Flag to keep node and relationship properties. By default set to true.
            write_triggers (bool) = True: Flag to export graph triggers.
            write_indexes (bool) = True: Flag to export indexes.
            write_constraints (bool) = True: Flag to export constraints.
    Returns:
        path (str): A path to the file where the query results are exported. If path is not provided, the output will be an empty string.
        data (str): A stream of query results in a cypher format.
    Raises:
        PermissionError: If you provided file path that you have no permissions to write at.
        OSError: If the file can't be opened or written to.
    """

    cypher = []

    memgraph = gqlalchemy.Memgraph()

    if config.get("write_triggers", True):
        triggers = memgraph.execute_and_fetch("SHOW TRIGGERS;")
        for trigger in triggers:
            cypher.append(
                f"CREATE TRIGGER {trigger['trigger name']} ON {trigger['event type']} {trigger['phase']} EXECUTE {trigger['statement']};"
            )
        cypher.append("")

    if config.get("write_indexes", True):
        constraints = memgraph.execute_and_fetch("SHOW CONSTRAINT INFO;")
        for constraint in constraints:
            constraint_type = constraint["constraint type"]

            if constraint_type == "exists":
                cypher.append(
                    f"CREATE CONSTRAINT ON (n:{constraint['label']}) ASSERT EXISTS (n.{constraint['properties']});"
                )
            elif constraint_type == "unique":
                properties = (
                    [constraint["properties"]]
                    if isinstance(constraint["properties"], str)
                    else list(constraint["properties"])
                )
                cypher.append(
                    f"CREATE CONSTRAINT ON (n:{constraint['label']}) ASSERT {'n.' + ', n.'.join(properties)} IS UNIQUE;"
                )
            else:
                raise ValueError("Unknown constraint type.")
        cypher.append("")

    if config.get("write_constraints", True):
        indexes = memgraph.execute_and_fetch("SHOW INDEX INFO;")
        for index in indexes:
            index_type = index["index type"]
            if index_type == "label":
                cypher.append(f"CREATE INDEX ON :{index['label']};")
            elif index_type == "label+property":
                cypher.append(
                    f"CREATE INDEX ON :{index['label']}({index['property']});"
                )
            else:
                raise ValueError("Unknown index type.")
        cypher.append("")

    graph = get_graph_for_cypher(ctx, config.get("write_properties", True))

    for object in graph:
        if isinstance(object, Node):
            object.labels.append("_IMPORT_ID")
            object.properties["_IMPORT_ID"] = object.id
            properties_str = (
                "{"
                + ", ".join([f"{k}: {v}" for k, v in object.properties.items()])
                + "}"
            )
            cypher.append(f"CREATE (n:{':'.join(object.labels)} {properties_str});")
        elif isinstance(object, Relationship):
            properties_str = (
                "{"
                + ", ".join([f"{k}: {v}" for k, v in object.properties.items()])
                + "}"
            )
            cypher.append(
                f"MATCH (n:_IMPORT_ID {{_IMPORT_ID: {object.start}}}) MATCH (m:_IMPORT_ID {{_IMPORT_ID: {object.end}}}) CREATE (n)-[:{object.label} {properties_str}]->(m);"
            )

    cypher.append("MATCH (n:_IMPORT_ID) REMOVE n:`_IMPORT_ID` REMOVE n._IMPORT_ID;")

    if path:
        try:
            with open(path, "w") as f:
                f.write("\n".join(cypher))
        except PermissionError:
            raise PermissionError(
                "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
            )
        except Exception:
            raise OSError("Could not open or write to the file.")

    return mgp.Record(
        path=path, data="\n".join(cypher) if config.get("stream", False) else ""
    )


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
