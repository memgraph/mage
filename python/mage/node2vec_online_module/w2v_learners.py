from gensim.models import Word2Vec
from typing import List, Any


class GensimWord2Vec:
    """
    gensim.Word2Vec wrapper for online representation learning

    Parameters
    ----------
    embedding_dimension : int
        Dimensions of the representation
    learning_rate : float
        Learning rate
    skip_gram: bool
        Use skip-gram model
    negative_rate: int
        Negative rate
    threads: int
        Maximum number of threads for parallelization
    """

    def __init__(
        self,
        embedding_dimension: int = 128,
        learning_rate: float = 0.01,
        skip_gram: bool = True,
        negative_rate: int = 10,
        threads: int = 4,
    ):
        self.embedding_dimension = embedding_dimension
        self.learning_rate = learning_rate
        self.skip_gram = skip_gram
        self.negative_rate = negative_rate
        self.threads = threads
        self.num_epochs = 1
        self.embeddings = None

    def partial_fit(self, sentences: List[List[Any]]) -> None:
        if self.model == None:
            if self.neg_rate < 0:
                self.model = Word2Vec(
                    sentences,
                    min_count=1,
                    vector_size=self.embedding_dimension,
                    window=1,
                    alpha=self.learning_rate,
                    min_alpha=self.learning_rate,
                    sg=int(self.skip_gram),
                    negative=0,
                    hs=1,
                    epochs=self.num_epochs,
                    workers=self.threads,
                )  # hierarchical softmax
            else:
                self.model = Word2Vec(
                    sentences,
                    min_count=1,
                    vector_size=self.embedding_dimension,
                    window=1,
                    alpha=self.learning_rate,
                    min_alpha=self.learning_rate,
                    sg=int(self.skip_gram),
                    negative=self.negative_rate,
                    epochs=self.num_epochs,
                    workers=self.threads,
                )
        # update model
        self.model.build_vocab(sentences, update=True)
        self.model.train(
            sentences, epochs=self.num_epochs, total_examples=self.model.corpus_count)
        self.embeddings = self.get_embedding_vectors()

    def get_embedding_vectors(self):
        vectors = self.model.wv.vectors
        indices = self.model.wv.index_to_key
        embeddings = {indices[i]: vectors[i] for i in range(len(indices))}
        return embeddings
