from typing import List
import mgp
import json as js


@mgp.read_proc
def json(ctx: mgp.ProcCtx, path: str) -> mgp.Record():
    """
    Procedure to export the whole database as JSON to a local file.

    Parameters
    ----------
    path : str
        Path to the file where JSON will be saved.
    """
    nodes = list()
    relationships = list()
    graph = list()

    for vertex in ctx.graph.vertices:
        labels = [label.name for label in vertex.labels]
        properties = dict()

        for key in vertex.properties.keys():
            properties[key] = vertex.properties.get(key)

        node = {
            "id": vertex.id,
            "labels": labels,
            "properties": properties,
            "type": "node",
        }
        nodes.append(node)

        for edge in vertex.out_edges:
            properties = dict()

            for key in edge.properties.keys():
                properties[key] = edge.properties.get(key)

            relationship = {
                "id": edge.id,
                "start": edge.from_vertex.id,
                "end": edge.to_vertex.id,
                "label": edge.type.name,
                "properties": properties,
                "type": "relationship",
            }
            relationships.append(relationship)

        graph = nodes + relationships

    with open(path, "w") as outfile:
        js.dump(graph, outfile, indent=4, sort_keys=True, default=str)

    return mgp.Record()
