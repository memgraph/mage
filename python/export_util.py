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
    id: int

    def get_dict(self) -> dict:
        return {
            Parameter.END.value: self.end,
            Parameter.ID.value: self.id,
            Parameter.LABEL.value: self.label,
            Parameter.PROPERTIES.value: self.properties,
            Parameter.START.value: self.start,
            Parameter.TYPE.value: Parameter.RELATIONSHIP.value,
        }


@dataclass
class KeyObjectGraphML:
    name: str
    is_for: str
    type: str
    type_is_list: bool
    default_value: str
    id: str = ""

    def __init__(
        self,
        name,
        is_for,
        type="",
        type_is_list=False,
        default_value="",
    ):
        self.name = name
        self.is_for = is_for
        self.type = type
        self.type_is_list = type_is_list
        self.default_value = default_value

    def __hash__(self):
        return hash(
            (
                self.name,
                self.is_for,
                self.type,
                self.type_is_list,
                self.default_value,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.name == other.name
            and self.is_for == other.is_for
            and self.type == other.type
            and self.type_is_list == other.type_is_list
            and self.default_value == other.default_value
        )


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


def to_duration_isoformat(value: timedelta) -> str:
    """Converts timedelta to ISO-8601 duration: P<date>T<time>"""
    date_parts: List[str] = []
    time_parts: List[str] = []

    if value.days != 0:
        date_parts.append(f"{abs(value.days)}D")

    if value.seconds != 0 or value.microseconds != 0:
        abs_seconds = abs(value.seconds)
        minutes, seconds = divmod(abs_seconds, 60)
        hours, minutes = divmod(minutes, 60)
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
        return to_duration_isoformat(property)

    if isinstance(property, (time, date, datetime)):
        return property.isoformat()

    else:
        return property


def get_graph(
    ctx: mgp.ProcCtx,
    config: Union[mgp.Map, None] = None,
) -> List[Union[Node, Relationship]]:
    """
    config : Map
        - graphML: bool
        - leaveOutLabels: bool
        - leaveOutProperties: bool

    """
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = []
        properties = dict()
        if not config.get("leaveOutLabels"):
            labels = [label.name for label in vertex.labels]
        if config.get("graphML") and not config.get("leaveOutProperties"):
            properties = {
                key: convert_to_isoformat_graphML(vertex.properties.get(key))
                for key in vertex.properties.keys()
            }
        elif not config.get("leaveOutProperties"):
            properties = {
                key: convert_to_isoformat(vertex.properties.get(key))
                for key in vertex.properties.keys()
            }

        nodes.append(Node(vertex.id, labels, properties).get_dict())

        for edge in vertex.out_edges:
            if not config.get("leaveOutProperties"):
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
            js.dump(
                graph,
                outfile,
                indent=Parameter.STANDARD_INDENT.value,
                default=str,
            )
    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."  # noqa: E501
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
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."  # noqa: E501
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
            "If you provided only stream value, it has to be set to True to get any results."  # noqa: E501
        )

    memgraph = Memgraph()
    results = list(memgraph.execute_and_fetch(query))

    # if query yields no result
    if not len(results):
        raise Exception(
            "Your query yields no results. Check if the database is empty or rewrite the provided query."  # noqa: E501
        )

    result_keys = list(results[0])
    data_list = [result_keys] + [list(result.values()) for result in results]
    data = ""

    if file_path:
        save_file(file_path, data_list)

    if stream:
        data = csv_to_stream(data_list)

    return mgp.Record(file_path=file_path, data=data)


def write_graphml_header(output: io.StringIO):
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


def check_if_elements_same_type(variable: List[Any]):
    if not isinstance(variable, (tuple, list)):
        return
    list_type = type(variable[0])
    for element in variable:
        if not isinstance(element, list_type):
            raise Exception(
                "If property value is a list it must consist of same typed elements."  # noqa: E501
            )


def get_type_string(variable: Any) -> Union[str, List[Any]]:
    if not isinstance(variable, tuple):
        return translate_types(variable), False
    if len(variable) == 0:
        return "string", True
    check_if_elements_same_type(variable)
    return translate_types(variable[0]), True


def write_key(
    output: io.StringIO,
    working_key: KeyObjectGraphML,
    key_id_counter: int,
    config: mgp.Map,
):
    output.write(
        f'<key id="d{key_id_counter}" for="{working_key.is_for}" attr.name="{working_key.name}"'      # noqa: E501
    )
    if config.get("useTypes"):
        if working_key.type_is_list:
            output.write(f' attr.type="string" attr.list="{working_key.type}"')
        else:
            output.write(f' attr.type="{working_key.type}"')
    output.write("/>\n")
    working_key.id = "d" + str(key_id_counter)


def get_gephi_label_value(
    element: Union[Node, Relationship], config: mgp.Map
) -> str:
    for caption in config.get("caption"):
        if caption in element.get("properties").keys():
            return str(element.get("properties").get(caption))

    if element.get("properties").values():
        return str(list(element.get("properties").values())[0])

    return str(element.get("id"))


def get_data_key(
    keys: set, name: str, is_for: str, type: str, is_list: bool = False
):
    for key in keys:
        if (
            key.name == name
            and key.is_for == is_for
            and key.type == type
            and key.type_is_list == is_list
        ):
            return key.id
    raise Exception(
        "This property doesn't have a key."  # noqa: E501 THIS SHOULD NOT HAPPEN ONCE I FIX GEPHI
    )


def write_labels_as_data(
    element: Union[Node, Relationship],
    output: io.StringIO,
    config: mgp.Map,
    keys: set,
):
    if not element.get("labels"):
        return

    if config.get("format").upper() == "GEPHI":
        output.write(
            f'<data key="{get_data_key(keys, "TYPE", "node", translate_types("TYPE"))}">'  # noqa: E501
        )
        for label in element.get("labels"):
            output.write(f":{label}")
        output.write("</data>")
        output.write(
            f'<data key="{get_data_key(keys, "labels", "node", translate_types("labels"))}">{get_gephi_label_value(element, config)}</data>'  # noqa: E501 SHOULD IT BE LABEL?
        )
        return

    if config.get("format").upper() == "TINKERPOP":
        output.write(
            f'<data key="{get_data_key(keys, "labelV", "node", translate_types("labelV"))}">'  # noqa: E501
        )
        for index, value in enumerate(element.get("labels")):
            if index == 0:
                output.write(value)
            else:
                output.write(f":{value}")
        output.write("</data>")
        return

    output.write(
        f'<data key="{get_data_key(keys, "labels", "node", translate_types("labels"))}">'  # noqa: E501
    )
    for label in element.get("labels"):
        output.write(f":{label}")
    output.write("</data>")


def get_value_string(value: Any) -> str:
    if isinstance(value, (set, list, tuple, map)):
        return js.dumps(value, ensure_ascii=False)
    return str(value)


def write_graphml_keys_nodes_rels(
    graph: List[Union[Node, Relationship]],
    keys_output: io.StringIO,
    nodes_and_rels_output: io.StringIO,
    config: mgp.Map,
) -> set:
    keys = set()
    key_id_counter = 0

    for element in graph:
        working_key = None
        if element.get("type") == "node":
            nodes_and_rels_output.write(f'<node id="n{str(element.get("id"))}')
            if (
                element.get("labels")
                and config.get("format").upper() != "TINKERPOP"
            ):
                nodes_and_rels_output.write('" labels="')
                for label in element.get("labels"):
                    nodes_and_rels_output.write(f":{label}")
            nodes_and_rels_output.write('">')

            if config.get("format").upper() == "GEPHI":
                working_key = KeyObjectGraphML(
                    "TYPE", "node", translate_types("TYPE")
                )
                keys.add(working_key)
                if len(keys) == key_id_counter + 1:  # something was added
                    write_key(keys_output, working_key, key_id_counter, config)
                    key_id_counter = key_id_counter + 1

            if element.get("labels"):
                if config.get("format").upper() == "TINKERPOP":
                    working_key = KeyObjectGraphML(
                        "labelV", "node", translate_types("labelV")
                    )
                else:  # SHOULD IT BE LABEL OR LABELS FOR GEPHI?
                    working_key = KeyObjectGraphML(
                        "labels", "node", translate_types("labels")
                    )
                keys.add(working_key)
                if len(keys) == key_id_counter + 1:  # something was added
                    write_key(keys_output, working_key, key_id_counter, config)
                    key_id_counter = key_id_counter + 1

            write_labels_as_data(element, nodes_and_rels_output, config, keys)

            for name, value in element.get("properties").items():
                type_string, is_list = get_type_string(value)
                working_key = KeyObjectGraphML(
                    name, "node", type_string, is_list
                )
                keys.add(working_key)
                if len(keys) == key_id_counter + 1:  # something was added
                    write_key(keys_output, working_key, key_id_counter, config)
                    key_id_counter = key_id_counter + 1
                else:
                    working_key.id = get_data_key(
                        keys, name, "node", type_string, is_list
                    )

                nodes_and_rels_output.write(
                    f'<data key="{working_key.id}">{get_value_string(value)}</data>'  # noqa: E501
                )
            nodes_and_rels_output.write("</node>\n")

        elif element.get("type") == "relationship":
            nodes_and_rels_output.write(
                f'<edge id="e{str(element.get("id"))}" source="n{str(element.get("start"))}" target="n{str(element.get("end"))}" label="{element.get("label")}">'  # noqa: E501
            )

            if config.get("format").upper() == "GEPHI":
                working_key = KeyObjectGraphML(
                    "TYPE", "edge", translate_types("TYPE")
                )
                keys.add(working_key)
                if len(keys) == key_id_counter + 1:  # something was added
                    write_key(keys_output, working_key, key_id_counter, config)
                    key_id_counter = key_id_counter + 1
                nodes_and_rels_output.write(
                    f'<data key="{get_data_key(keys, "TYPE", "edge", translate_types("TYPE"))}">{element.get("label")}</data>'  # noqa: E501
                )
            if config.get("format").upper() == "TINKERPOP":
                working_key = KeyObjectGraphML(
                    "labelE", "edge", translate_types("labelE")
                )
            else:
                working_key = KeyObjectGraphML(
                    "label", "edge", translate_types("label")
                )
            keys.add(working_key)
            if len(keys) == key_id_counter + 1:  # something was added
                write_key(keys_output, working_key, key_id_counter, config)
                key_id_counter = key_id_counter + 1
            nodes_and_rels_output.write(
                f'<data key="{get_data_key(keys, working_key.name, "edge", working_key.type)}">{element.get("label")}</data>'  # noqa: E501
            )

            for name, value in element.get("properties").items():
                type_string, is_list = get_type_string(value)
                working_key = KeyObjectGraphML(
                    name, "edge", type_string, is_list
                )
                keys.add(working_key)
                if len(keys) == key_id_counter + 1:  # something was added
                    write_key(keys_output, working_key, key_id_counter, config)
                    key_id_counter = key_id_counter + 1
                else:
                    working_key.id = get_data_key(
                        keys, name, "edge", type_string, is_list
                    )

                nodes_and_rels_output.write(
                    f'<data key="{working_key.id}">{get_value_string(value)}</data>'  # noqa: E501
                )
            nodes_and_rels_output.write("</edge>\n")


def write_graphml_graph_id(output: io.StringIO):
    output.write('<graph id="G" edgedefault="directed">\n')


def write_graphml_footer(output: io.StringIO):
    output.write("</graph>\n")
    output.write("</graphml>")


def set_default_config(config: mgp.Map) -> mgp.Map:
    if config is None:
        config = dict()
    if not config.get("stream"):
        config.update({"stream": False})
    if not config.get("format"):
        config.update({"format": ""})
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
        or not isinstance(config.get("caption"), tuple)
        or not isinstance(config.get("useTypes"), bool)
        or not isinstance(config.get("leaveOutLabels"), bool)
        or not isinstance(config.get("leaveOutProperties"), bool)
    ):
        raise TypeError(
            "Config parameter must be a map with specific keys and values described in documentation."  # noqa: E501
        )
    return config


@mgp.read_proc
def graphml(
    ctx: mgp.ProcCtx,
    path: str = "",
    config: Union[mgp.Map, None] = None,
) -> mgp.Record(status=str):
    """
    Procedure to export the whole database to a graphML file.

    Parameters
    ----------
    path : str
        Path to the graphML file containing the exported graph database.
    config : Map

    """

    config = set_default_config(config)
    graph_config = {"graphML": True}
    graph_config.update({"leaveOutLabels": config.get("leaveOutLabels")})
    graph_config.update(
        {"leaveOutProperties": config.get("leaveOutProperties")}
    )

    graph = get_graph(ctx, graph_config)

    if not path and not config.get("stream"):
        raise Exception(
            "Please provide file name or set stream to True in config."
        )

    output = io.StringIO()
    keys_output = io.StringIO()
    nodes_and_rels_output = io.StringIO()

    write_graphml_header(output)
    write_graphml_keys_nodes_rels(
        graph, keys_output, nodes_and_rels_output, config
    )
    output.write(keys_output.getvalue())
    write_graphml_graph_id(output)
    output.write(nodes_and_rels_output.getvalue())
    write_graphml_footer(output)

    try:
        if path:
            with open(path, "w") as outfile:
                outfile.write(output.getvalue())
            outfile.close()
    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."  # noqa: E501
        )
    except Exception:
        raise OSError("Could not open or write to the file.")

    if config.get("stream"):
        return mgp.Record(status=output.getvalue())

    return mgp.Record(status="success")
