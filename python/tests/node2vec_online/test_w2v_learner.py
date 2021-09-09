import pytest

from mage.node2vec_online_module.w2v_learners import GensimWord2Vec


@pytest.fixture
def w2v_learner():
    return GensimWord2Vec(
        embedding_dimension=2,
        learning_rate=0.01,
        skip_gram=True,
        negative_rate=0,
        threads=1,
    )


def test_calculate_embeddings(w2v_learner):
    sentences = [[1, 2], [2, 4], [3, 2]]
    w2v_learner.partial_fit(sentences)
    embeddings_dict = w2v_learner.get_embedding_vectors()
    assert len(embeddings_dict) == 4
