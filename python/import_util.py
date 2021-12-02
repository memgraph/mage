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
    list = js.load(file)
    vertex_ids = dict()

    for el in list:
        if el["type"] == "node":
            vertex = ctx.graph.create_vertex()
            vertex_properties = vertex.properties
            for key, value in el["properties"].items():
                vertex_properties[key] = value
            for label in el["labels"]:
                vertex.add_label(label)
            vertex_ids[el["id"]] = vertex.id
        elif el["type"] == "relationship":
            vertex_from = ctx.graph.get_vertex_by_id(vertex_ids[el["start"]])
            vertex_to = ctx.graph.get_vertex_by_id(vertex_ids[el["end"]])
            edge = ctx.graph.create_edge(vertex_from, vertex_to, mgp.EdgeType(el["label"]))
            edge_properties = edge.properties
            for key, value in el["properties"].items():
                edge_properties[key] = value
    return mgp.Record()
