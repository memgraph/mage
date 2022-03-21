from datetime import datetime, date, time, timedelta
import json as js
from mage.export_import_util.parameters import Parameter
import mgp
from typing import Union, List, Dict, Any


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


def create_vertex(ctx: mgp.ProcCtx, properties: Dict[str, Any], labels: List[str]):
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
                "Each graph object needs to have 'type', 'properties' and 'id' keys."
            )

        if type_value == Parameter.NODE.value:

            if Parameter.LABELS.value in object:
                labels_value = object[Parameter.LABELS.value]
            else:
                raise KeyError("Each node object needs to have 'labels' key.")

            vertex_ids[id_value] = create_vertex(ctx, properties_value, labels_value)

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
                    "Each relationship object needs to have 'start', 'end' and 'label' keys."
                )

            create_edge(
                ctx, properties_value, start_node_id, end_node_id, edge_type, vertex_ids
            )
        else:
            raise KeyError("The provided file does not match the correct JSON format.")

    return mgp.Record()
