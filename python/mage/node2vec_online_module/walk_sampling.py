import random
import numpy as np


class StreamWalkUpdater:
    """
    Sample temporal random walks for the StreamWalk algorithm

    Parameters
    ----------
    half_life : int
        Half-life in seconds for time decay
    max_len : int
        Maximum length of the sampled temporal random walks
    beta : float
        Damping factor for long paths
    cutoff: int
        Temporal cutoff in seconds to exclude very distant past
    k: int
        Number of sampled walks for each edge update
    full_walks: bool
        Return every node of the sampled walk for representation learning (full_walks=True) or only the endpoints of the walk (full_walks=False)
    """

    def __init__(
        self, half_life=7200, max_len=3, beta=0.9, cutoff=604800, k=4, full_walks=False
    ):
        self.c = -np.log(0.5) / half_life
        self.beta = beta
        self.half_life = half_life
        self.k = k
        self.cutoff = cutoff
        self.max_len = max_len
        self.full_walks = full_walks
        self.G = {}
        self.times = {}
        self.cent = {}
        self.cent_now = {}
        self.lens = {}
        for j in range(max_len):
            self.lens[j + 1] = 0

    def __str__(self):
        return "streamwalk_hl%i_ml%i_beta%.2f_cutoff%i_k%i_fullw%s" % (
            self.half_life,
            self.max_len,
            self.beta,
            self.cutoff,
            self.k,
            self.full_walks,
        )

    def process_new_edge(self, src, trg, time):
        self.update(src, trg, time)
        return self.sample_node_pairs(src, trg, time, self.k)

    def sample_node_pairs(self, src, trg, time, sample_num):
        if src not in self.G:
            # src is not reachable from any node within cutoff
            return [(src, trg)] * sample_num
        edge_tuples = [(src, trg, time)] * sample_num
        pairs = [self.sample_single_walk(tup) for tup in edge_tuples]
        return pairs

    def sample_single_walk(self, edge_tuple):
        src, trg, time = edge_tuple
        node_, time_, cent_ = src, self.times[src], self.cent[src]
        walk = []
        walk.append(node_)
        while True:
            if (
                random.uniform(0, 1) < 1 / (cent_ * self.beta + 1)
                or (node_ not in self.G)
                or len(walk) >= self.max_len
            ):
                break
            sum_ = cent_ * random.uniform(0, 1)
            sum__ = 0
            broken = False
            for (n, t, c) in reversed(self.G[node_]):
                if t < time_:
                    sum__ += (c * self.beta + 1) * np.exp(self.c * (t - time_))
                    if sum__ >= sum_:
                        broken = True
                        break
            if not broken:
                break
            node_, time_, cent_ = n, t, c
            walk.append(node_)
        self.lens[len(walk)] += 1
        if self.full_walks:
            return [trg] + walk
        else:
            return (node_, trg)

    def update(self, src, trg, time):
        # apply time decay for trg
        if trg in self.cent:
            self.cent[trg] = self.cent[trg] * np.exp(self.c * (self.times[trg] - time))
        else:
            self.cent[trg] = 0
        src_cent = 0
        if src in self.times:
            src_cent = self.cent[src]
            if self.times[src] < time:
                src_cent = src_cent * np.exp(self.c * (self.times[src] - time))
                # update centrality and time for src
                self.cent[src] = src_cent
                self.times[src] = time
                self.cent_now[src] = 0
                self.clean_in_edges(src, time)
            else:
                # if src is currently active then adjust centrality
                src_cent = src_cent - self.cent_now[src]
        self.cent[trg] += src_cent * self.beta + 1
        if (trg not in self.times) or (self.times[trg] < time):
            # cent_now is initialized for each node in each second
            self.cent_now[trg] = 0
        self.cent_now[trg] += src_cent * self.beta + 1
        # collect recent edges for each vertex
        if trg not in self.G:
            self.G[trg] = []
        self.G[trg].append((src, time, src_cent))
        self.times[trg] = time
        # clean in egdes
        self.clean_in_edges(trg, time)

    def clean_in_edges(self, node, time):
        ii = 0
        for (s, t, c) in self.G[node]:
            if time - t < self.cutoff:
                break
            ii += 1
        # drop old inedges
        self.G[node] = self.G[node][ii:]
