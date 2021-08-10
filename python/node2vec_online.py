import multiprocessing
import time

import mgp

from mage.node2vec_online_module.w2v_learners import (GensimWord2Vec)
from mage.node2vec_online_module.walk_sampling import (StreamWalkUpdater)


class Node2VecContext:
    def __init__(self):
        self._updater = None
        self._learner = None
        self._start_time = None

    @property
    def updater(self):
        return self._updater

    @property
    def learner(self):
        return self._learner

    @property
    def start_time(self):
        return self._start_time

    @updater.setter
    def updater(self, updater):
        self._updater = updater

    @learner.setter
    def learner(self, learner):
        self._learner = learner

    @start_time.setter
    def start_time(self, start_time):
        self._start_time = start_time

    def is_initialized(self):
        return (
            self._learner is not None
            and self._updater is not None
            and self._start_time is not None
        )


context = Node2VecContext()


def update_model(source: int, target: int, current_time: int):
    global context
    sampled_pairs = context.updater.process_new_edge(source, target, current_time)
    context.learner.partial_fit(sampled_pairs)


@mgp.read_proc
def set(
    ctx: mgp.ProcCtx,
    half_life: int = 7200,
    max_length: int = 3,
    beta: mgp.Number = 0.9,
    cuttoff: int = 604800,
    sampled_walks: int = 4,
    full_walks: bool = False,
    embedding_dimension: int = 128,
    learning_rate: mgp.Number = 0.01,
    skip_gram: bool = True,
    negative_rate: mgp.Number = 10,
    threads: mgp.Nullable[int] = None,
):

    global context

    if threads is None:
        threads = multiprocessing.cpu_count()

    current_time = time.time()

    if not context.is_initialized():
        context.start_time = current_time
        context.updater = StreamWalkUpdater(
            half_life=half_life,
            max_length=max_length,
            beta=beta,
            cutoff=cuttoff,
            sampled_walks=sampled_walks,
            full_walks=full_walks,
        )

        context.learner = GensimWord2Vec(
            embedding_dimension=embedding_dimension,
            learning_rate=learning_rate,
            skip_gram=skip_gram,
            negative_rate=negative_rate,
            threads=threads,
        )

    return mgp.Record()


@mgp.read_proc
def get(
    ctx: mgp.ProcCtx,
) -> mgp.Record(node=mgp.Vertex, embedding=mgp.List[mgp.Number]):

    if not context.is_initialized():
        return mgp.Record()

    embeddings_ = context.learner.get_embedding_vectors()

    embeddings = {}
    for node_id, embedding in embeddings_.items():
        embeddings[node_id] = [float(e) for e in embedding]

    return [
        mgp.Record(node=ctx.graph.get_vertex_by_id(node_id), embedding=e)
        for node_id, e in embeddings.items()
    ]


@mgp.read_proc
def update(ctx: mgp.ProcCtx, edges: mgp.List[mgp.Edge]):
    global context

    current_time = time.time()
    if context.is_initialized():
        for e in edges:
            ctx.check_must_abort()
            update_model(e.from_vertex.id, e.to_vertex.id, current_time)

    return mgp.Record()
