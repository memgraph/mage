from datetime import datetime, date, time, timedelta
import json as js
from mage.export_import_util.parameters import Parameter
import mgp
from typing import Union, List, Dict, Any
import re


def convert_from_isoformat(
    property: Union[None, str, bool, int, float, List[Any], Dict[str, Any]]
):
    if not isinstance(property, str):
        return property

    if str.startswith(property, Parameter.DURATION.value):
        duration_iso = property.split("(")[-1].split(")")[0]
        parsed_time = datetime.strptime(duration_iso, "%H:%M:%S.%f")
        return timedelta(
            hours=parsed_time.hour,
            minutes=parsed_time.minute,
            seconds=parsed_time.second,
            microseconds=parsed_time.microsecond,
        )
    elif str.startswith(property, Parameter.LOCALTIME.value):
        local_time_iso = property.split("(")[-1].split(")")[0]
        return time.fromisoformat(local_time_iso)
    elif str.startswith(property, Parameter.LOCALDATETIME.value):
        local_datetime_iso = property.split("(")[-1].split(")")[0]
        return datetime.fromisoformat(local_datetime_iso)
    elif str.startswith(property, Parameter.DATE.value):
        date_iso = property.split("(")[-1].split(")")[0]
        return date.fromisoformat(date_iso)
    else:
        return property


def create_vertex(
    ctx: mgp.ProcCtx, properties: Dict[str, Any], labels: List[str]
):
    vertex = ctx.graph.create_vertex()
    vertex_properties = vertex.properties

    for key, value in properties.items():
        vertex_properties[key] = convert_from_isoformat(value)

    for label in labels:
        vertex.add_label(label)

    return vertex.id


def create_edge(
    ctx: mgp.ProcCtx,
    properties: Dict[str, Any],
    start_node_id: int,
    end_node_id: int,
    type: str,
    vertex_ids: Dict[int, int],
):
    vertex_from = ctx.graph.get_vertex_by_id(vertex_ids[start_node_id])
    vertex_to = ctx.graph.get_vertex_by_id(vertex_ids[end_node_id])
    edge = ctx.graph.create_edge(vertex_from, vertex_to, mgp.EdgeType(type))
    edge_properties = edge.properties

    for key, value in properties.items():
        edge_properties[key] = convert_from_isoformat(value)


@mgp.write_proc
def json(ctx: mgp.ProcCtx, path: str) -> mgp.Record():
    """
    Procedure to import the JSON created by the export_util.json procedure.

    Parameters
    ----------
    path : str
        Path to the JSON file that is being imported.
    """
    try:
        with open(path, "r") as file:
            graph_objects = js.load(file)
    except Exception:
        raise OSError("Could not open/read file.")

    vertex_ids = dict()

    for object in graph_objects:
        if all(
            key in object
            for key in (
                Parameter.TYPE.value,
                Parameter.PROPERTIES.value,
                Parameter.ID.value,
            )
        ):
            type_value = object[Parameter.TYPE.value]
            properties_value = object[Parameter.PROPERTIES.value]
            id_value = object[Parameter.ID.value]
        else:
            raise KeyError(
                "Each graph object needs to have 'type', \
                 'properties' and 'id' keys."
            )

        if type_value == Parameter.NODE.value:
            if Parameter.LABELS.value in object:
                labels_value = object[Parameter.LABELS.value]
            else:
                raise KeyError("Each node object needs to have 'labels' key.")

            vertex_ids[id_value] = create_vertex(
                ctx, properties_value, labels_value
            )

        elif type_value == Parameter.RELATIONSHIP.value:
            if all(
                key in object
                for key in (
                    Parameter.START.value,
                    Parameter.END.value,
                    Parameter.LABEL.value,
                )
            ):
                start_node_id = object[Parameter.START.value]
                end_node_id = object[Parameter.END.value]
                edge_type = object[Parameter.LABEL.value]
            else:
                raise KeyError(
                    "Each relationship object needs to have 'start', \
                     'end' and 'label' keys."
                )

            create_edge(
                ctx,
                properties_value,
                start_node_id,
                end_node_id,
                edge_type,
                vertex_ids,
            )
        else:
            raise KeyError(
                "The provided file does not match the correct JSON format."
            )

    return mgp.Record()


def set_default_config(config):
    if not config.get("readLabels"):
        config.update({"readLabels": False})
    if not config.get("defaultRelationshipType"):
        config.update({"defaultRelationshipType": "RELATED"})
    if not config.get("storeNodeIds"):
        config.update({"storeNodeIds": False})
    if not config.get("source"):
        config.update({"source": {}})
    if not config.get("target"):
        config.update({"target": {}})


def get_attribute(line, attribute_name):
    match = re.search(fr'{attribute_name}="(\w+)"', line)
    if match:
        return match.group(1)
    return None


@mgp.write_proc
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

    set_default_config(config)

    try:
        with open(path, "r") as file:
            file_content = file.read()
            file_content = file_content.replace('\t', '')
            file_content = file_content.replace('><', '>\n<')
            file_content = file_content.replace('\n<data', '<data')
            # file_content = file_content.replace('\n</node', '</node')
            # file_content = file_content.replace('\n</edge', '</edge')
    except Exception:
        raise OSError("Could not open/read file.")

    node_keys = dict()
    rel_keys = dict()

    for line in file_content:

        if (line.startswith("<?xml") or line.startswith("<graph") or line.startswith("<graphml")):
            continue
        if (line.startswith("</")):
            continue
        if (line.startswith("<key")):
            if (get_attribute(line, "for") == "node"):
                node_keys.update({get_attribute(line, "id"): [get_attribute(line, "name"), get_attribute(line, "type"), get_attribute(line, "list")]})
            if (get_attribute(line, "for") == "relationship"):
                rel_keys.update({get_attribute(line, "id"): [get_attribute(line, "name"), get_attribute(line, "type"), get_attribute(line, "list")]})

    print(file_content)

    file.close()

    return mgp.Record(status="success")
