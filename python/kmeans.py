"""
Purpose of this query module is to offer easy kmeans clustering algorithm on top of the embeddings that you
might have stored in nodes. All you need to do is call kmeans.get_clusters(5, "embedding") where 5
represents number of clusters you want to get, and "embedding" represents node property name in which
embedding of node is stored
"""
from sklearn.cluster import KMeans
from typing import List, Tuple

import mgp


def get_created_clusters(number_of_clusters: int, embeddings: List[List[float]], nodes: List[mgp.Vertex]) -> List[
    Tuple[mgp.Vertex, int]]:
    kmeans = KMeans(n_clusters=number_of_clusters).fit(embeddings)
    return [(nodes[i], label) for i, label in enumerate(kmeans.labels_)]


@mgp.write_proc
def get_clusters(
        ctx: mgp.ProcCtx, number_of_groups: mgp.Number, embedding_property: str = "embedding") -> mgp.Record(
    vertex=mgp.Vertex, cluster=mgp.Number):
    nodes = []
    embeddings = []
    for node in ctx.graph.vertices:
        nodes.append(node)
        embeddings.append(node.properties.get(embedding_property))

    nodes_labels_list = get_created_clusters(number_of_groups, embeddings, nodes)
    return [mgp.Record(vertex=vertex, cluster=int(label)) for vertex, label in nodes_labels_list]


@mgp.write_proc
def set_clusters(
        ctx: mgp.ProcCtx, number_of_groups: mgp.Number, embedding_property: str = "embedding",
        label_property="label") -> mgp.Record( vertex=mgp.Vertex, cluster=mgp.Number):
    nodes = []
    embeddings = []
    for node in ctx.graph.vertices:
        nodes.append(node)
        embeddings.append(node.properties.get(embedding_property))

    nodes_labels_list = get_created_clusters(number_of_groups, embeddings, nodes)

    for vertex, label in nodes_labels_list:
        vertex.properties.set(label_property, int(label))

    return [mgp.Record(vertex=vertex, cluster=int(label)) for vertex, label in nodes_labels_list]
