import csv
import io
import json as js
import mgp

from math import floor
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
                f".{abs(value.microseconds)}"
                if value.microseconds != 0
                else ""
            )
            time_parts.append(f"{seconds}{microseconds_part}S")

    date_duration_str = "".join(date_parts)
    time_duration_str = f'T{"".join(time_parts)}' if time_parts else ""

    return f"P{date_duration_str}{time_duration_str}"


def convert_to_isoformat_graphML(
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
        return to_duration_iso_format(property)

    if isinstance(property, (time, date, datetime)):
        return property.isoformat()

    else:
        return property


def get_graph(
    ctx: mgp.ProcCtx, graphML: bool
) -> List[Union[Node, Relationship]]:
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = [label.name for label in vertex.labels]
        if graphML:
            properties = {
                key: convert_to_isoformat_graphML(vertex.properties.get(key))
                for key in vertex.properties.keys()
            }
        else:
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

    graph = get_graph(ctx, 0)
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
            "You don't have permissions to write into that file. Make sure \
             to give the necessary permissions to user memgraph."
        )
    except Exception:
        raise OSError("Could not open or write to the file.")

    return mgp.Record()


@mgp.read_proc
def json_stream(ctx: mgp.ProcCtx) -> mgp.Record(stream=str):
    """
    Procedure to export the whole database to a stream.
    """
    return mgp.Record(stream=js.dumps(get_graph(ctx, 0)))


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
            "You don't have permissions to write into that file. \
             Make sure to give the necessary permissions to user memgraph."
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
    """  # noqa: E501

    # file or config have to be provided
    if not file_path and not stream:
        raise Exception("Please provide file name and/or config.")

    # only config provided with stream set to false
    if not file_path and not stream:
        raise Exception(
            "If you provided only stream value, \
             it has to be set to True to get any results."
        )

    memgraph = Memgraph()
    results = list(memgraph.execute_and_fetch(query))

    # if query yields no result
    if not len(results):
        raise Exception(
            "Your query yields no results. Check if the database \
             is empty or rewrite the provided query."
        )

    result_keys = list(results[0])
    data_list = [result_keys] + [list(result.values()) for result in results]
    data = ""

    if file_path:
        save_file(file_path, data_list)

    if stream:
        data = csv_to_stream(data_list)

    return mgp.Record(file_path=file_path, data=data)


def write_header(output: io.StringIO):
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write(
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n'  # noqa: E501
    )


def translate_types(variable: Any):
    if isinstance(variable, str):
        return "string"
    if isinstance(variable, bool):
        return "boolean"
    if isinstance(variable, float):
        return "float"
    if isinstance(variable, int):
        return "int"
    raise Exception(
        "Property values can only be primitive types or arrays of primitive types."  # noqa: E501
    )


def check_if_elements_same_type(variable: Any):
    for element in variable:
        if not isinstance(element, type(variable[0])):
            raise Exception(
                "If property value is a list it must consist of same typed elements."  # noqa: E501
            )


def get_type_string(variable: Any) -> Union[str, List[Any]]:
    if (
        isinstance(variable, tuple)
    ):
        if len(variable) == 0:
            return ["string", "string"]
        check_if_elements_same_type(variable)
        return [
            "string",
            translate_types(variable[0]),
        ]
    return translate_types(variable)


def write_keys(
    graph: List[Union[Node, Relationship]],
    output: io.StringIO,
    config: mgp.Map,
):
    node_keys = dict()
    rel_keys = dict()
    for element in graph:
        if element.get("type") == "node":
            if element.get("labels"):
                if config.get("format").upper() == "TINKERPOP":
                    node_keys.update({"labelV": get_type_string("labelV")})
                else:
                    node_keys.update(
                        {"labels": get_type_string("labels")}
                    )  # what if two nodes have same property name but different value type? or node's property name is "labels"?  # noqa: E501

            for key, value in element.get("properties").items():
                node_keys.update({key: get_type_string(value)})

        elif element.get("type") == "relationship":
            if config.get("format").upper() == "TINKERPOP":
                rel_keys.update({"labelE": get_type_string("labelE")})
            else:
                rel_keys.update({"label": get_type_string("label")})

            for key, value in element.get("properties").items():
                rel_keys.update({key: get_type_string(value)})

    if config.get("format").upper() == "GEPHI":
        node_keys.update({"TYPE": get_type_string("TYPE")})
        rel_keys.update({"TYPE": get_type_string("TYPE")})

    for key, value in node_keys.items():
        output.write(f'<key id="{key}" for="node" attr.name="{key}"')
        if config.get("useTypes"):
            if isinstance(value, str):
                output.write(f' attr.type="{value}"')
            else:
                output.write(f' attr.type="{value[0]}" attr.list="{value[1]}"')
        output.write("/>\n")
    for key, value in rel_keys.items():
        output.write(f'<key id="{key}" for="edge" attr.name="{key}"')
        if config.get("useTypes"):
            if isinstance(value, str):
                output.write(f' attr.type="{value}"')
            else:
                output.write(f' attr.type="{value[0]}" attr.list="{value[1]}"')
        output.write("/>\n")


def write_graph(output: io.StringIO):
    output.write('<graph id="G" edgedefault="directed">\n')


def get_gephi_label_value(
    element: Union[Node, Relationship], config: mgp.Map
) -> str:
    for caption in config.get("caption"):
        if caption in element.get("properties").keys():
            return str(element.get("properties").get(caption))

    if element.get("properties").values():
        return str(list(element.get("properties").values())[0])

    return str(element.get("id"))


def write_labels_as_data(
    element: Union[Node, Relationship], output: io.StringIO, config: mgp.Map
):
    if not element.get("labels"):
        return
    if config.get("format").upper() == "TINKERPOP":
        output.write('<data key="labelV">')
        for i in range(0, len(element.get("labels"))):
            if i == 0:
                output.write(element.get("labels")[i])
            else:
                output.write(f':{element.get("labels")[i]}')
        output.write("</data>")
    if config.get("format").upper() == "GEPHI":
        output.write('<data key="TYPE">')
        for label in element.get("labels"):
            output.write(f":{label}")
        output.write("</data>")
        output.write(
            f'<data key="label">{get_gephi_label_value(element, config)}</data>'  # noqa: E501
        )
    else:
        output.write('<data key="labels">')
        for label in element.get("labels"):
            output.write(f":{label}")
        output.write("</data>")


def get_value_string(value: Any) -> str:
    if isinstance(value, (set, list, tuple, map)):
        return js.dumps(value, ensure_ascii=False)
    return str(value)


def write_nodes_and_rels(
    graph: List[Union[Node, Relationship]],
    output: io.StringIO,
    config: mgp.Map,
):
    for element in graph:
        if element.get("type") == "node":
            output.write(f'<node id="n{str(element.get("id"))}')
            if (
                element.get("labels")
                and config.get("format").upper() != "TINKERPOP"
            ):
                output.write('" labels="')
                for label in element.get("labels"):
                    output.write(f":{label}")
            output.write('">')

            write_labels_as_data(element, output, config)

            for key, value in element.get("properties").items():
                output.write(
                    f'<data key="{key}">{get_value_string(value)}</data>'
                )
            output.write("</node>\n")

        elif element.get("type") == "relationship":
            output.write(
                f'<edge id="e{str(element.get("id"))}" source="n{str(element.get("start"))}" target="n{str(element.get("end"))}" label="{element.get("label")}">'  # noqa: E501
            )

            if config.get("format").upper() == "TINKERPOP":
                output.write(
                    f'<data key="labelE">{element.get("label")}</data>'
                )
            else:
                output.write(
                    f'<data key="label">{element.get("label")}</data>'
                )
            if config.get("format").upper() == "GEPHI":
                output.write(f'<data key="TYPE">{element.get("label")}</data>')

            for key, value in element.get("properties").items():
                output.write(
                    f'<data key="{key}">{get_value_string(value)}</data>'
                )
            output.write("</edge>\n")


def write_footer(output: io.StringIO):
    output.write("</graph>\n")
    output.write("</graphml>")


def set_default_config(config: mgp.Map):
    if not config.get("stream"):
        config.update({"stream": False})
    if not config.get("format"):
        config.update({"format": ""})  # should it be Gephi?
    if not config.get("caption"):
        config.update({"caption": []})
    if not config.get("useTypes"):
        config.update({"useTypes": False})
    if not config.get("leaveOutLabels"):
        config.update({"leaveOutLabels": False})
    if not config.get("leaveOutProperties"):
        config.update({"leaveOutProperties": False})
    if (
        not isinstance(config.get("stream"), bool)
        or not isinstance(config.get("format"), str)
        or not isinstance(config.get("caption"), list)
        or not isinstance(config.get("useTypes"), bool)
        or not isinstance(config.get("leaveOutLabels"), bool)
        or not isinstance(config.get("leaveOutProperties"), bool)
    ):
        raise TypeError(
            "Config parameter must be a map with specific \
             keys and values described in documentation."
        )


@mgp.read_proc
def graphml(
    ctx: mgp.ProcCtx,
    path: str = "",
    config: mgp.Map = {},
) -> mgp.Record(status=str):
    """
    Procedure to export the whole database to a graphML file.

    Parameters
    ----------
    path : str
        Path to the graphML file containing the exported graph database.
    config : Map

    """

    graph = get_graph(ctx, 1)
    set_default_config(config)

    if config.get("leaveOutLabels") or config.get("leaveOutProperties"):
        for element in graph:
            if config.get("leaveOutLabels"):
                element.update({"labels": []})
            if config.get("leaveOutProperties"):
                element.update({"properties": {}})

    output = io.StringIO()

    if not path and not config.get("stream"):
        raise Exception(
            "Please provide file name or set stream to True in config."
        )

    write_header(output)
    write_keys(graph, output, config)
    write_graph(output)
    write_nodes_and_rels(graph, output, config)
    write_footer(output)

    try:
        if path:
            with open(path, "w") as outfile:
                outfile.write(output.getvalue())
            outfile.close()
    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure \
             to give the necessary permissions to user memgraph."
        )
    except Exception:
        raise OSError("Could not open or write to the file.")

    if config.get("stream"):
        return mgp.Record(status=output.getvalue())

    return mgp.Record(status="success")
