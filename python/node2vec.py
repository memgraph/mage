from typing import List, Dict

import gensim
import mgp

from mage.node2vec.second_order_random_walk import SecondOrderRandomWalk
from mage.node2vec.graph import BasicGraph

word2vec_params = {
    "min_count": 1,
    "vector_size": 64,
    "window": 1,
    "alpha": 0.1,
    "min_alpha": 1,
    "sg": 1,
    "epochs": 10,
    "workers": 1,
}


def learn_embeddings(
    walks: List[List[int]], **word2vec_params
) -> Dict[int, List[float]]:
    model = gensim.models.Word2Vec(sentences=walks, **word2vec_params)

    vectors = model.wv.vectors
    indices = model.wv.index_to_key
    embeddings = {indices[i]: vectors[i] for i in range(len(indices))}

    return embeddings


@mgp.read_proc
def set_word2vec_params(
    ctx: mgp.ProcCtx,
    vector_size=100,
    alpha=0.025,
    window=5,
    min_count=5,
    seed=1,
    workers=3,
    min_alpha=0.0001,
    sg=1,
    hs=0,
    negative=5,
    epochs=5,
) -> mgp.Record(
    vector_size=int,
    window=int,
    min_count=int,
    workers=int,
    min_alpha=float,
    seed=int,
    alpha=float,
    epochs=int,
    sg=int,
    negative=int,
    hs=int,
):
    """
    Function to set parameters used in gensim.models.Word2Vec

    Parameters
    ----------
    vector_size : int, optional
        Dimensionality of the word vectors.
    window : int, optional
        Maximum distance between the current and predicted word within a sentence.
    min_count : int, optional
        Ignores all words with total frequency lower than this.
    workers : int, optional
        Use these many worker threads to train the model (=faster training with multicore machines).
    sg : {0, 1}, optional
        Training algorithm: 1 for skip-gram; otherwise CBOW.
    hs : {0, 1}, optional
        If 1, hierarchical softmax will be used for model training.
        If 0, and `negative` is non-zero, negative sampling will be used.
    negative : int, optional
        If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
        should be drawn (usually between 5-20).
        If set to 0, no negative sampling is used.
    cbow_mean : {0, 1}, optional
        If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    alpha : float, optional
        The initial learning rate.
    min_alpha : float, optional
        Learning rate will linearly drop to `min_alpha` as training progresses.
    seed : int, optional
        Seed for the random number generator. Initial vectors for each word are seeded with a hash of
        the concatenation of word + `str(seed)`.
    """
    word2vec_params["vector_size"] = vector_size
    word2vec_params["window"] = window
    word2vec_params["min_count"] = min_count
    word2vec_params["workers"] = workers
    word2vec_params["min_alpha"] = min_alpha
    word2vec_params["seed"] = seed
    word2vec_params["alpha"] = alpha
    word2vec_params["epochs"] = epochs
    word2vec_params["sg"] = sg
    word2vec_params["negative"] = negative
    word2vec_params["hs"] = hs

    return mgp.Record(
        vector_size=word2vec_params["vector_size"],
        window=word2vec_params["window"],
        min_count=word2vec_params["min_count"],
        workers=word2vec_params["workers"],
        min_alpha=word2vec_params["min_alpha"],
        seed=word2vec_params["seed"],
        alpha=word2vec_params["alpha"],
        epochs=word2vec_params["epochs"],
        sg=word2vec_params["sg"],
        negative=word2vec_params["negative"],
        hs=word2vec_params["hs"],
    )


@mgp.read_proc
def get_embeddings(
    ctx: mgp.ProcCtx,
    edges: List[mgp.Edge],
    is_directed: bool = False,
    p=1,
    q=1,
    num_walks=4,
    walk_length=5,
) -> mgp.Record(node=mgp.Number, embedding=mgp.List[mgp.Number]):
    edges_weights = {}

    for edge in edges:
        edge_weight = float(edge.properties.get("weight", default=1))
        src_id = int(edge.from_vertex.properties.get("id", edge.from_vertex.id))
        dest_id = int(edge.to_vertex.properties.get("id", edge.to_vertex.id))
        edges_weights[(src_id, dest_id)] = edge_weight

    graph = BasicGraph(edges_weights, is_directed)
    second_order_random_walk = SecondOrderRandomWalk(
        p=int(p), q=int(q), num_walks=int(num_walks), walk_length=int(walk_length)
    )
    walks = second_order_random_walk.sample_node_walks(graph)
    print("walks", walks)
    embeddings = learn_embeddings(walks, **word2vec_params)

    for node_id, embedding in embeddings.items():
        embeddings[node_id] = [float(e) for e in embedding]
        print(node_id, embeddings[node_id])
    return [
        mgp.Record(node=int(node_id), embedding=embedding)
        for node_id, embedding in embeddings.items()
    ]
