import time
import multiprocessing
from mage.node2vec_online_module import GensimWord2Vec, StreamWalkUpdater

MAX_THREADS = multiprocessing.cpu_count()


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

    @property
    def elapsed_time(self) -> int:
        return time.time() - self._start_time

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
        return self._learner is None or self._updater is None or self._start_time is None


context = Node2VecContext()


def init_context(
    start_time,
    half_life: int,
    max_length: int,
    beta: float,
    cuttoff: int,
    k: int,
    full_walks: bool,
    embedding_dims: int,
    learning_rate: float,
    sg: bool,
    negative_rate: float,
    threads: int,
):
    global context
    context.start_time = start_time
    context.updater = StreamWalkUpdater(
        half_life=half_life,
        max_len=max_length,
        beta=beta,
        cutoff=cuttoff,
        k=k,
        full_walks=full_walks,
    )
    context.learner = GensimWord2Vec(
        embedding_dims=embedding_dims,
        lr_rate=learning_rate,
        sg=sg,
        neg_rate=negative_rate,
        n_threads=threads,
    )


def update_model(source, target, current_time):
    global context
    sampled_pairs = self.updater.process_new_edge(source, target, current_time)
    context.learner.partial_fit(sampled_pairs, current_time)


@mgp.read_proc
def get(
    ctx: mgp.ProcCtx,
    half_life: int = 7200,
    max_length: int = 3,
    beta: float = 0.9,
    cuttoff: int = 604800,
    k: int = 4,
    full_walks: bool = False,
    embedding_dims: int = 128,
    learning_rate: float = 0.01,
    sg: bool = True,
    negative_rate: float = 10,
    threads: int = MAX_THREADS,
) -> mgp.Record(node=mgp.Vertex, embedding=mgp.List[float]):
    global context

    current_time = time.time()
    if not context.is_initialized():
        init_context(
            current_time,
            half_life,
            max_length,
            beta,
            cuttoff,
            k,
            full_walks,
            embedding_dims,
            learning_rate,
            sg,
            negative_rate,
            threads,
        )

    embeddingd = context.learner.get_embedding_vectors()

    return [
        mgp.Record(node=node, embedding=embeddings[node]) for node in ctx.graph.nodes
    ]


@mgp.read_proc
def update(
    ctx: mgp.ProcCtx,
    edges: mgp.List[mgp.Edge],
    half_life: int = 7200,
    max_length: int = 3,
    beta: float = 0.9,
    cuttoff: int = 604800,
    k: int = 4,
    full_walks: bool = False,
    embedding_dims: int = 128,
    learning_rate: float = 0.01,
    sg: bool = True,
    negative_rate: float = 10,
    threads: int = MAX_THREADS,
):
    global context

    current_time = time.time()
    if not context.is_initialized():
        init_context(
            current_time,
            half_life,
            max_length,
            beta,
            cuttoff,
            k,
            full_walks,
            embedding_dims,
            learning_rate,
            sg,
            negative_rate,
            threads,
        )

    for e in edges:
        ctx.check_must_abort()
        update_model(e.from_vertex.id, e.to_vertex.id, current_time)
