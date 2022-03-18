from datetime import datetime, date, time, timedelta
import json as js
from mage.export_import_util.parameters import Parameter
import mgp


def convert_from_isoformat(property):
    if isinstance(property, str):
        if str.startswith(property, Parameter.DURATION.value):
            parsed_time = datetime.strptime(
                property.split("(")[-1].split(")")[0], "%H:%M:%S.%f"
            )
            return timedelta(
                hours=parsed_time.hour,
                minutes=parsed_time.minute,
                seconds=parsed_time.second,
                microseconds=parsed_time.microsecond,
            )
        if str.startswith(property, Parameter.LOCALTIME.value):
            return time.fromisoformat(property.split("(")[-1].split(")")[0])
        if str.startswith(property, Parameter.LOCALDATETIME.value):
            return datetime.fromisoformat(property.split("(")[-1].split(")")[0])
        if str.startswith(property, Parameter.DATE.value):
            return date.fromisoformat(property.split("(")[-1].split(")")[0])

    return property


def create_vertex(ctx, properties, labels):
    vertex = ctx.graph.create_vertex()
    vertex_properties = vertex.properties

    for key, value in properties.items():
        vertex_properties[key] = convert_from_isoformat(value)

    for label in labels:
        vertex.add_label(label)

    return vertex.id


def create_edge(ctx, properties, start_node_id, end_node_id, type, vertex_ids):
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
