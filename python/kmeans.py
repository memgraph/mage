import time
from typing import List, Tuple

import mgp

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def get_labels(nodes, embeddings, number_of_clusters):
    classes_embeddings_classes_dict, classes_embeddings_list = get_groups(number_of_clusters, embeddings, nodes)
    return classes_embeddings_classes_dict


def get_groups(number_of_clusters, embeddings, nodes) -> List[Tuple[mgp.Vertex, int]]:
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=int(time.time())).fit(embeddings_scaled)

    kmeans_labels = kmeans.labels_

    classes_embedding_list = []
    for i in range(len(kmeans_labels)):
        label = kmeans_labels[i]
        classes_embedding_list.append((nodes[i], label))

    return classes_embedding_list


@mgp.write_proc
def set_labels(
        ctx: mgp.ProcCtx, nodes: mgp.List[mgp.Vertex], embeddings: mgp.List[mgp.List[mgp.Number]],
        number_of_groups: mgp.Number, label_property="label") -> mgp.Record():
    nodes_new = []
    for node in nodes:
        nodes_new.append(node)

    embeddings_new = []
    for embedding in embeddings:
        embeddings_new.append([float(e) for e in embedding])

    nodes_labels_list = get_groups(number_of_groups, embeddings_new, nodes_new)

    for vertex, label in nodes_labels_list:
        vertex.properties.set(label_property, int(label))

    return mgp.Record()
