from typing import List, Dict

import gensim
import mgp

from mage.node2vec.second_order_random_walk import SecondOrderRandomWalk
from mage.node2vec.graph import BasicGraph


def learn_embeddings(walks: List[List[int]],  **skip_gram_params) -> Dict[int, List[float]]:

    model = gensim.models.Word2Vec(walks, **skip_gram_params)

    vectors = model.wv.vectors
    indices = model.wv.index_to_key
    embeddings = {indices[i]: vectors[i] for i in range(len(indices))}


    return embeddings


@mgp.read_proc
def calculate_embeddings(ctx: mgp.ProcCtx, edges: List[mgp.Edge], is_directed: bool = False) -> mgp.Record():
    edges_weights = {}

    for edge in edges:
        edge_weight = float(edge.properties.get('weight', default=1))
        src_id = int(edge.from_vertex.properties.get('id', edge.from_vertex.id))
        dest_id = int(edge.to_vertex.properties.get('id', edge.to_vertex.id))
        edges_weights[(src_id, dest_id)] = edge_weight

    graph = BasicGraph(edges_weights, is_directed)
    second_order_random_walk = SecondOrderRandomWalk(p=1, q=2, num_walks=1, walk_length=1)
    walks = second_order_random_walk.sample_node_walks(graph)

    params = {
        "min_count": 1,
        "vector_size": 64,
        "window": 1,
        "alpha": 0.1,
        "min_alpha": 1,
        "sg": True,
        "epochs": 10,
        "workers": 1,
    }
    learn_embeddings(walks, **params)

    return mgp.Record()
