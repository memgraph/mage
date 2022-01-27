import mgp
import json as js
import sys


@mgp.write_proc
def json(ctx: mgp.ProcCtx, path: str) -> mgp.Record():
    """
    Procedure to import the local JSON created by the export_util.json procedure.

    Parameters
    ----------
    path : str
        Path to the JSON that is being imported.
    """
    file = open(path)
    graph_objects = js.load(file)
    vertex_ids = dict()

    for object in graph_objects:

        try:
            type_value = object["type"]
            properties_value = object["properties"]
            id_value = object["id"]
        except KeyError as key_error:
            sys.stderr.write(
                (f"Each graph object needs to have 'type', 'properties' and 'id' keys.")
            )
            raise key_error

        if type_value == "node":
            try:
                labels_value = object["labels"]
            except KeyError as key_error:
                sys.stderr.write((f"Each node object needs to have 'labels' key."))
                raise key_error

            vertex = ctx.graph.create_vertex()
            vertex_properties = vertex.properties

            for key, value in properties_value.items():
                vertex_properties[key] = value

            for label in labels_value:
                vertex.add_label(label)

            vertex_ids[id_value] = vertex.id

        elif type_value == "relationship":
            try:
                start_node_id = object["start"]
                end_node_id = object["end"]
                edge_type = object["label"]
            except KeyError as key_error:
                sys.stderr.write(
                    (
                        f"Each relationship object needs to have 'start', 'end' and 'label' keys."
                    )
                )
                raise key_error

            vertex_from = ctx.graph.get_vertex_by_id(vertex_ids[start_node_id])
            vertex_to = ctx.graph.get_vertex_by_id(vertex_ids[end_node_id])
            edge = ctx.graph.create_edge(
                vertex_from, vertex_to, mgp.EdgeType(edge_type)
            )
            edge_properties = edge.properties

            for key, value in properties_value.items():
                edge_properties[key] = value
        else:
            raise KeyError("The provided file is not in the correct JSON form.")

    return mgp.Record()
