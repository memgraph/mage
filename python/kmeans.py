from sklearn.cluster import KMeans
from typing import List, Tuple

import mgp


def get_created_clusters(number_of_clusters: int, embeddings: List[List[float]], nodes: List[mgp.Vertex]) -> List[
    Tuple[mgp.Vertex, int]]:
    kmeans = KMeans(n_clusters=number_of_clusters).fit(embeddings)
    return [(nodes[i], label) for i, label in enumerate(kmeans.labels_)]


@mgp.write_proc
def get_clusters(
        ctx: mgp.ProcCtx, nodes: mgp.List[mgp.Vertex], embeddings: mgp.List[mgp.List[mgp.Number]],
        number_of_groups: mgp.Number) -> mgp.Record(vertex=mgp.Vertex, cluster=mgp.Number):
    nodes_labels_list = get_created_clusters(number_of_groups, embeddings, nodes)
    return [mgp.Record(vertex=vertex, label=label) for vertex, label in nodes_labels_list]



@mgp.write_proc
def set_clusters(
        ctx: mgp.ProcCtx, nodes: mgp.List[mgp.Vertex], embeddings: mgp.List[mgp.List[mgp.Number]],
        number_of_groups: mgp.Number, label_property="label") -> mgp.Record():
 
    nodes_labels_list = get_created_clusters(number_of_groups, embeddings, nodes)

    for vertex, label in nodes_labels_list:
        vertex.properties.set(label_property, int(label))

    return mgp.Record()
