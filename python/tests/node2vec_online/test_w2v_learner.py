import pytest

from mage.node2vec_online_module.w2v_learners import GensimWord2Vec

EMBEDDINGS_DIM = 2
INCORRECT_NEGATIVE_RATE = -1


@pytest.fixture
def w2v_learner():
    return GensimWord2Vec(
        embedding_dimension=EMBEDDINGS_DIM,
        learning_rate=0.01,
        skip_gram=True,
        negative_rate=0,
        threads=1,
    )


@pytest.fixture
def w2v_learner_wrong_negative_rate():
    return GensimWord2Vec(
        embedding_dimension=EMBEDDINGS_DIM,
        learning_rate=0.01,
        skip_gram=True,
        negative_rate=INCORRECT_NEGATIVE_RATE,
        threads=1,
    )


def test_calculate_embeddings(w2v_learner):
    sentences = [[1, 2], [2, 4], [3, 2]]
    w2v_learner.partial_fit(sentences)
    embeddings_dict = w2v_learner.get_embedding_vectors()
    assert len(embeddings_dict) == 4


def test_correct_embedding_dimension(w2v_learner):
    sentences = [[1, 2], [2, 4], [3, 2]]
    w2v_learner.partial_fit(sentences)
    embeddings_dict = w2v_learner.get_embedding_vectors()
    for key, value in embeddings_dict.items():
        assert len(value) == EMBEDDINGS_DIM


def test_incorrect_negative_rate(w2v_learner_wrong_negative_rate):
    assert w2v_learner_wrong_negative_rate.negative_rate == INCORRECT_NEGATIVE_RATE
    assert w2v_learner_wrong_negative_rate.embedding_dimension == EMBEDDINGS_DIM

    sentences = [[1, 2], [2, 4], [3, 2]]
    w2v_learner_wrong_negative_rate.partial_fit(sentences)

    assert w2v_learner_wrong_negative_rate.negative_rate == 0


def test_correct_training(w2v_learner):
    sentences = [[1, 2], [2, 4], [3, 2]]

    w2v_learner.partial_fit(sentences)
    calculated_embeddings_dict = w2v_learner.get_embedding_vectors()

    non_existing_sentence = [[3, 4]]
    w2v_learner.partial_fit(non_existing_sentence)

    new_embeddings_dict = w2v_learner.get_embedding_vectors()

    for key, value in calculated_embeddings_dict.items():
        assert key in new_embeddings_dict
        all([a == b for a, b in zip(value, new_embeddings_dict[key])])
