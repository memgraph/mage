from datetime import datetime, date, time, timedelta
import json as js
from mage.export_import_util.parameters import Parameter
import mgp
from typing import Union, List, Dict, Any
import defusedxml.ElementTree as ET
from export_util import convert_to_isoformat_graphML
import ast


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


def find_node(
    ctx: mgp.ProcCtx, label: str, prop_key: str, prop_value: str
) -> Any:  # TODO prop value cast??
    for vertex in ctx.graph.vertices:
        if (
            label in [label.name for label in vertex.labels]
            and prop_key in vertex.properties.keys()
            and convert_to_isoformat_graphML(vertex.properties.get(prop_key))
            == prop_value
        ):
            return vertex.id
    return None


def cast_element(
    text: str, type: str
) -> Union[List[Any], str, int, bool, float]:
    if text is None:
        return ""
    if type == "string":
        return str(text)
    if type == "int":
        return int(text)
    if type == "boolean":
        return bool(text)
    if type == "float":
        return float(text)
    if type is None:
        return text


def cast(
    text: str, type: str, is_list: str
) -> Union[List[Any], str, int, bool, float]:
    if is_list is not None:
        casted_list = list()
        for element in ast.literal_eval(text):
            casted_list.append(cast_element(element, is_list))
        return casted_list
    return cast_element(text, type)


def set_default_keys(key_dict: Dict[str, Any], properties: Dict[str, Any]):
    for key, value in key_dict.items():
        if value[3] is not None:
            properties.update({value[0]: cast(value[3], value[1], value[2])})


def set_default_config(config: mgp.Map):
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
    if (
        not isinstance(config.get("readLabels"), bool)
        or not isinstance(config.get("defaultRelationshipType"), str)
        or not isinstance(config.get("storeNodeIds"), bool)
        or not isinstance(config.get("source"), dict)
        or not isinstance(config.get("target"), dict)
        or (
            config.get("source") and "label" not in config.get("source").keys()
        )
        or (
            config.get("target") and "label" not in config.get("target").keys()
        )
    ):
        raise TypeError(
            "Config parameter must be a map with specific \
             keys and values described in documentation."
        )


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
        tree = ET.parse(path)
    except Exception:
        raise OSError("Could not open/read file.")

    root = tree.getroot()
    graphml_ns = root.tag.split("}")[0].strip("{")
    namespace = {"graphml": graphml_ns}

    node_keys = dict()
    rel_keys = dict()

    for key in root.findall(".//graphml:key", namespace):
        # key value legend: value[0] = attr.name
        #                   value[1] = attr.type
        #                   value[2] = attr.list
        #                   value[3] = default
        # -> should I make a class out of it?
        value = [key.attrib["attr.name"]]
        if "attr.type" in key.attrib.keys():
            value.append(key.attrib["attr.type"])
        else:
            value.append(None)
        if "attr.list" in key.attrib.keys():
            value.append(key.attrib["attr.list"])
        else:
            value.append(None)
        child = key.findall(".//default")
        if child:
            value.append(child[0].text)
        else:
            value.append(None)
        if key.attrib["for"].lower() == "node":
            node_keys.update({key.attrib["id"]: value})
        elif key.attrib["for"].lower() == "edge":
            rel_keys.update({key.attrib["id"]: value})

    real_ids = dict()

    for node in root.findall(".//graphml:node", namespace):
        labels = []
        properties = dict()
        if config.get("readLabels"):
            labels = node.attrib["labels"].split(":")
            labels.pop(0)
        if config.get("storeNodeIds"):
            properties.update({"id": node.attrib["id"]})

        set_default_keys(node_keys, properties)

        for data in node.findall("graphml:data", namespace):
            key = node_keys.get(data.attrib["key"])
            if key is None:
                key = [data.attrib["key"], "string", None, None]
            if config.get("readLabels") and data.attrib["key"] == "labels":
                new_labels = node.attrib["labels"].split(":")
                new_labels.pop(0)
                labels = labels + new_labels
            else:
                properties.update({key[0]: cast(data.text, key[1], key[2])})

        real_ids.update(
            {node.attrib["id"]: create_vertex(ctx, properties, labels)}
        )

    for rel in root.findall(".//graphml:edge", namespace):
        if "label" in rel.attrib.keys():
            rel_type = rel.attrib["label"]
        else:
            rel_type = config.get("defaultRelationshipType")

        properties = dict()
        set_default_keys(rel_keys, properties)

        for data in rel.findall("graphml:data", namespace):
            key = rel_keys.get(data.attrib["key"])
            if key is None:
                key = [data.attrib["key"], "string", None, None]
            if not data.attrib["key"] == "label":  # Tinkerpop???
                properties.update({key[0]: cast(data.text, key[1], key[2])})

        if rel.attrib["source"] not in real_ids.keys():
            if not config.get("source"):
                # without source/target config, we look for the internal id
                real_ids.update({rel.attrib["source"]: rel.attrib["source"]})
            else:
                source_config = config.get("source")
                if "id" not in source_config.keys():
                    source_config.update({"id": "id"})
                node_id = find_node(
                    ctx,
                    source_config["label"],
                    source_config["id"],
                    rel.attrib["source"],
                )
                real_ids.update({rel.attrib["source"]: node_id})

        if rel.attrib["target"] not in real_ids.keys():
            if not config.get("target"):
                # without source/target config, we look for the internal id
                real_ids.update({rel.attrib["target"]: rel.attrib["target"]})
            else:
                target_config = config.get("target")
                if "id" not in target_config.keys():
                    target_config.update({"id": "id"})
                node_id = find_node(
                    ctx,
                    target_config["label"],
                    target_config["id"],
                    rel.attrib["target"],
                )
                real_ids.update({rel.attrib["target"]: node_id})

        create_edge(
            ctx,
            properties,
            rel.attrib["source"],
            rel.attrib["target"],
            rel_type,
            real_ids,
        )

    return mgp.Record(status="success")