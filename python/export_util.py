from dataclasses import dataclass
import json as js
import mgp
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
        properties = {
            key: vertex.properties.get(key) for key in vertex.properties.keys()
        }

        nodes.append(Node(vertex.id, labels, properties).get_dict())

        for edge in vertex.out_edges:
            properties = dict()
            properties = {
                key: edge.properties.get(key) for key in edge.properties.keys()
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

        graph = nodes + relationships

    try:
        with open(path, "w") as outfile:
            js.dump(graph, outfile, indent=4, default=str)
    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the user memgraph the necessary permissions"
        )
    except Exception:
        raise OSError("Could not open or write to the file.")

    return mgp.Record()
