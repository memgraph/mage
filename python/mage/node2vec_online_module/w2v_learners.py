from gensim.models import Word2Vec
import numpy as np
from collections import Counter, deque


class GensimWord2Vec:
    """
    gensim.Word2Vec wrapper for online representation learning

    Parameters
    ----------
    embedding_dims : int
        Dimensions of the representation
    lr_rate : float
        Learning rate
    sg: 0/1
        Use skip-gram model
    neg_rate: int
        Negative rate
    n_threads: int
        Maximum number of threads for parallelization
    """

    def __init__(
        self, embedding_dims=128, lr_rate=0.01, sg=True, neg_rate=10, n_threads=4
    ):
        self.embedding_dims = embedding_dims
        self.lr_rate = lr_rate
        self.sg = sg
        self.neg_rate = neg_rate
        self.n_threads = n_threads
        self.num_epochs = 1
        self.embeddings = None

    def __str__(self):
        return "gensimw2v_dim%i_lr%0.4f_neg%i_sg%i" % (
            self.embedding_dims,
            self.lr_rate,
            self.neg_rate,
            self.sg,
        )

    def partial_fit(self, sentences):
        if self.model == None:
            if self.neg_rate < 0:
                self.model = Word2Vec(
                    sentences,
                    min_count=1,
                    vector_size=self.embedding_dims,
                    window=1,
                    alpha=self.lr_rate,
                    min_alpha=self.lr_rate,
                    sg=int(self.sg),
                    negative=0,
                    hs=1,
                    epochs=self.num_epochs,
                    workers=self.n_threads,
                )  # hierarchical softmax
            else:
                self.model = Word2Vec(
                    sentences,
                    min_count=1,
                    vector_size=self.embedding_dims,
                    window=1,
                    alpha=self.lr_rate,
                    min_alpha=self.lr_rate,
                    sg=int(self.sg),
                    negative=self.neg_rate,
                    epochs=self.num_epochs,
                    workers=self.n_threads,
                )
        # update model
        self.model.build_vocab(sentences, update=True)
        self.model.train(
            sentences, epochs=self.num_epochs, total_words=len(self.all_words)
        )
        self.embeddings = self.get_embedding_vectors()

    def get_embedding_vectors(self):
        vectors = self.model.wv.vectors
        indices = self.model.wv.index_to_key
        embeddings = {indices[i]: vectors[i] for i in range(len(indices))}
        return embeddings
