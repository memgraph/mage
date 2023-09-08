import csv
import io
import json as js
import mgp
import gqlalchemy
import os

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from gqlalchemy import Memgraph
from math import floor

from typing import Any, Dict, List, Union
from math import floor

from mage.export_import_util.parameters import Parameter

HEADER_FILENAME = "header.csv"


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


def get_properties_json(object, write_properties: bool):
    return (
        {
            key: convert_to_isoformat(object.properties.get(key))
            for key in object.properties.keys()
        }
        if write_properties
        else {}
    )


def get_graph(
    ctx: mgp.ProcCtx, write_properties: bool
) -> List[Union[Node, Relationship]]:
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = [label.name for label in vertex.labels]
        properties = get_properties_json(vertex, write_properties)

        nodes.append(Node(vertex.id, labels, properties).get_dict())

        for edge in vertex.out_edges:
            properties = get_properties_json(edge, write_properties)

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


def get_graph_from_list(
    graph_vertices: list, graph_edges: list, write_properties: bool
) -> List[Union[Node, Relationship]]:
    nodes = list()
    relationships = list()

    for vertex in graph_vertices:
        labels = [label.name for label in vertex.labels]
        properties = get_properties_json(vertex, write_properties)

        nodes.append(Node(vertex.id, labels, properties).get_dict())

    for edge in graph_edges:
        properties = get_properties_json(edge, write_properties)

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


def get_graph_info_from_lists(
    node_list: List[mgp.Vertex], relationship_list: List[mgp.Edge]
):
    graph = list()
    all_node_properties = list()
    all_node_prop_set = set()
    all_relationship_properties = list()
    all_relationship_prop_set = set()

    for node in node_list:
        for prop in node.properties:
            if prop not in all_node_prop_set:
                all_node_properties.append(prop)
                all_node_prop_set.add(prop)
        graph.append(Node(node.id, node.labels, node.properties))
    all_node_properties.sort()

    for relationship in relationship_list:
        for prop in relationship.properties:
            if prop not in all_relationship_prop_set:
                all_relationship_properties.append(prop)
                all_relationship_prop_set.add(prop)

        graph.append(
            Relationship(
                relationship.to_vertex.id,
                relationship.id,
                relationship.type.name,
                relationship.properties,
                relationship.from_vertex.id,
            )
        )
    all_relationship_properties.sort()

    return graph, all_node_properties, all_relationship_properties


def json_dump_to_file(graph: List[Union[Node, Relationship]], path: str):
    try:
        with open(path, "w") as outfile:
            js.dump(
                graph,
                outfile,
                indent=Parameter.STANDARD_INDENT.value,
                default=str,
            )
    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file."
            "Make sure to give the necessary permissions to user memgraph."
        )
    except Exception:
        raise OSError("Could not open or write to the file.")


@mgp.read_proc
def json(
    ctx: mgp.ProcCtx, path: str = "", config: mgp.Map = {}
) -> mgp.Record(path=str, data=str):
    """
    Procedure to export the whole database to a JSON file.

    Parameters:
        context : mgp.ProcCtx
            Reference to the context execution.
        path : str = ""
            Path to the JSON file containing the exported graph database.
        config : mgp.Map
            stream (bool) = False: Flag to export the graph data to a stream.
            write_properties (bool) = True: Flag to keep node and relationship properties. By default set to true.

    Returns:
        path (str): A path to the file where the query results are exported. If path is not provided, the output will be an empty string.
        data (str): A stream of query results in JSON format.

    Raises:
        PermissionError: If you provided file path that you have no permissions to write at.
        OSError: If the file can't be opened or written to.
    """
    graph = get_graph(ctx, config.get("write_properties", True))
    if path:
        json_dump_to_file(graph, path)

    return mgp.Record(
        path=path,
        data=js.dumps(graph) if config.get("stream", False) else "",
    )


@mgp.read_proc
def json_graph(
    ctx: mgp.ProcCtx,
    nodes: list,
    relationships: list,
    path: str = "",
    config: mgp.Map = {},
) -> mgp.Record(path=str, data=str):
    """
    Procedure to export the given graph to a JSON file. The graph is given with a map that contains keys "nodes" and "relationships".

    Parameters:
        nodes : List[Node]
            A list thats contains all nodes in the given graph.
        relationships : List[Relationship]
            A list that containts all the relationships in the given graph.
        path : str
            Path to the JSON file containing the exported graph database.
        config : mgp.Map
            stream (bool) = False: Flag to export the graph data to a stream.
            write_properties (bool) = True: Flag to keep node and relationship properties. By default set to true.

    Returns:
        path (str): A path to the file where the query results are exported. If path is not provided, the output will be an empty string.
        data (str): A stream of query results in JSON format.

    Raises:
        PermissionError: If you provided file path that you have no permissions to write at.
        OSError: If the file can't be opened or written to.
    """
    graph = get_graph_from_list(
        nodes, relationships, config.get("write_properties", True)
    )
    if path:
        json_dump_to_file(graph, path)

    return mgp.Record(
        path=path,
        data=js.dumps(graph) if config.get("stream", False) else "",
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
            "You don't have permissions to write into that file."
            "Make sure to give the necessary permissions to user memgraph."
        )
    except csv.Error as e:
        raise csv.Error(
            "Could not write to the file {}, stopped at line {}: {}".format(
                file_path, writer.line_num, e
            )
        )
    except Exception:
        raise OSError("Could not open or write to the file.")


def csv_to_stream(
    data_list: list, delimiter: str = ",", quoting_type=csv.QUOTE_NONNUMERIC
) -> str:
    output = io.StringIO()
    try:
        writer = csv.writer(
            output, delimiter=delimiter, quoting=quoting_type, escapechar="\\"
        )
        writer.writerows(data_list)
    except csv.Error as e:
        raise csv.Error(
            "Could not write a stream, stopped at line {}: {}".format(
                writer.line_num, e
            )
        )
    return output.getvalue()


def csv_header(
    node_properties: List[str], relationship_properties: List[str]
) -> List[str]:
    """
    This function creates the header for csv file
    """
    header = ["_id", "_labels"]

    for prop in node_properties:
        header.append(prop)

    header.extend(["_start", "_end", "_type"])

    for prop in relationship_properties:
        header.append(prop)

    return [header]


def process_properties(
    properties: Dict[str, mgp.Any], prop: str, write_list: List[mgp.Any]
) -> None:
    if isinstance(properties[prop], (set, list, tuple, map)):
        write_list.append(js.dumps(properties[prop]))
        return

    if isinstance(properties[prop], timedelta):
        write_list.append(convert_to_isoformat(properties[prop]))
        return

    write_list.append(properties[prop])


def csv_data_list(
    graph: List[Union[Node, Relationship]],
    node_properties: List[str],
    relationship_properties: List[str],
) -> List[mgp.Any]:
    """
    Function that parses graph into a data_list appropriate for csv writing
    """
    data_list = []
    for element in graph:
        write_list = []
        is_node = isinstance(element, Node)

        # processing id and labels part
        if is_node:
            write_list.extend(
                [
                    element.id,
                    "".join(":" + label.name for label in element.labels),
                ]
            )
        else:
            write_list.extend(["", ""])

        # node_properties
        for prop in node_properties:
            if prop in element.properties and is_node:
                process_properties(element.properties, prop, write_list)
            else:
                write_list.append("")
        # relationship
        if is_node:
            # start, end, type
            write_list.extend(["", "", ""])
        else:
            # start, end, type
            write_list.extend([element.start, element.end, element.label])

        # relationship properties
        for prop in relationship_properties:
            if prop in element.properties and not is_node:
                process_properties(element.properties, prop, write_list)
            else:
                write_list.append("")

        data_list.append(write_list)

    return data_list


def check_config_valid(config: mgp.Any, type: mgp.Any, name: str):
    if not isinstance(config, type):
        raise TypeError("Config attribute {0} must be of type {1}".format(name, type))


def csv_process_config(config: mgp.Map):
    delimiter = ","
    if "delimiter" in config:
        check_config_valid(config["delimiter"], str, "delimiter")

        delimiter = config["delimiter"]

    quoting_type = csv.QUOTE_ALL
    if "quotes" in config:
        check_config_valid(config["quotes"], str, "quotes")

        if config["quotes"] == "none":
            quoting_type = csv.QUOTE_NONE
        elif config["quotes"] == "ifNeeded":
            quoting_type = csv.QUOTE_MINIMAL

    separate_header = False
    if "separateHeader" in config:
        check_config_valid(config["separateHeader"], bool, "separateHeader")
        separate_header = config["separateHeader"]

    stream = False
    if "stream" in config:
        check_config_valid(config["stream"], bool, "stream")
        stream = config["stream"]

    return delimiter, quoting_type, separate_header, stream


def header_path(path: str):
    directory, filename = os.path.split(path)
    new_filename = HEADER_FILENAME
    return os.path.join(directory, new_filename)


# todo: remove later when we figure what to do with it
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


def write_file(path: str, delimiter: str, quoting_type: str, data: mgp.Any) -> None:
    with open(path, "w") as file:
        writer = csv.writer(
            file, delimiter=delimiter, quoting=quoting_type, escapechar="\\"
        )
        writer.writerows(data)


@mgp.read_proc
def csv_graph(
    nodes_list: mgp.List[mgp.Vertex],
    relationships_list: mgp.List[mgp.Edge],
    path: str = "",
    config: mgp.Map = {},
) -> mgp.Record(path=str, data=str):
    """
    Procedure to export the given graph to a csv file.
    The graph is given with two lists, one for nodes,
    and one for relationships.


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

        delimiter (string) = ,: Delimiter for csv file.

        quotes (string) = always : Option which quoting type to use

        separateHeader (bool) = False: Flag to separate header into another
        csv file

    """
    if path == "":
        path = "exported_file.csv"
    delimiter, quoting_type, separate_header, stream = csv_process_config(config)
    (
        graph,
        node_properties,
        relationship_properties,
    ) = get_graph_info_from_lists(nodes_list, relationships_list)
    data_list = csv_data_list(graph, node_properties, relationship_properties)
    header = csv_header(node_properties, relationship_properties)

    try:
        if separate_header:
            if not stream:
                write_file(header_path(path), delimiter, quoting_type, header)
        else:
            data_list = header + data_list

        if stream:
            data = csv_to_stream(data_list, delimiter, quoting_type)
            return mgp.Record(path=path, data=data)

        write_file(path, delimiter, quoting_type, data_list)

    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file."
            "Make sure to give the necessary permissions to user memgraph."
        )
    except Exception:
        raise OSError("Could not open or write to the file.")
    return mgp.Record(
        path=path,
        data="",
    )


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
        query (str): A query from which the results will be
        saved to a CSV file.

        file_path (str, optional): A path to the CSV file where the query
        results will be exported. Defaults to an empty string.

        stream (bool, optional): A value which determines whether a
        stream of query results in a CSV format will be returned.
    Returns:
        mgp.Record(
            file_path (str): A path to the CSV file where the query results are
            exported. If file_path is not provided, the output will be an
            empty string.
            data (str): A stream of query results in a CSV format.
        )
    Raises:
        Exception: If neither file nor config are provided,
        or if only config is provided with stream set to False.
        Also if query yields no results or if the database is empty.
        PermissionError: If you provided file path that you have
        no permissions to write at.
        csv.Error: If an error occurred while writing into stream or CSV file.
        OSError: If the file can't be opened or written to.
    """

    # file or config have to be provided
    if not file_path and not stream:
        raise Exception("Please provide file name and/or config.")

    # only config provided with stream set to false
    if not file_path and not stream:
        raise Exception(
            "If you provided only stream value, it has to be set to"
            "True to get any results."
        )

    memgraph = Memgraph()
    results = list(memgraph.execute_and_fetch(query))

    # if query yields no result
    if not len(results):
        raise Exception(
            "Your query yields no results."
            "Check if the database is empty or rewrite the provided query."
        )

    result_keys = list(results[0])
    data_list = [result_keys] + [list(result.values()) for result in results]
    data = ""

    if file_path:
        save_file(file_path, data_list)

    if stream:
        data = csv_to_stream(data_list)

    return mgp.Record(file_path=file_path, data=data)
