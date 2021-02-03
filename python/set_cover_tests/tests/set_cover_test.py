import pytest
from database import Memgraph, Node
from query_module import delete_all, create_node, insert_nodes, insert_relationship, set_cover
from itertools import chain


SET = 'set'
ELEMENT = 'element'


def get_quoted_string(string):
    return f"'{string}'"


def compare(s, t):
    return sorted(s, key=lambda ele: sorted(ele.items())) == sorted(t, key=lambda ele: sorted(ele.items()))




@pytest.fixture
def db():
    db = Memgraph()
    return db


@pytest.fixture
def db_delete():
    db = Memgraph()
    delete_all(db)


@pytest.fixture(scope='session', autouse=True)
def db_delete_after():
    yield
    db = Memgraph()
    delete_all(db)


# Test suite 1
def test_result_is_present(db, db_delete):
    e1 = create_node(node_id=1, properties={'group': get_quoted_string(ELEMENT)})
    s1 = create_node(node_id=2, properties={'group': get_quoted_string(SET)})

    e2 = create_node(node_id=3, properties={'group' : get_quoted_string(ELEMENT)})
    s2 = create_node(node_id=4, properties={'group': get_quoted_string(SET)})

    e3 = create_node(node_id=5, properties={'group': get_quoted_string(ELEMENT)})
    s3 = create_node(node_id=6, properties={'group': get_quoted_string(SET)})

    insert_nodes(db, [e1, e2, e3, s1, s2, s3])
    insert_relationship(db, e1, s1)
    insert_relationship(db, e2, s2)
    insert_relationship(db, e3, s3)

    resulting_sets = set_cover(db)
    assert resulting_sets is not None


# Test suite 2
def test_greedy_result_is_present(db, db_delete):
    e1 = create_node(node_id=1, properties={'group': get_quoted_string(ELEMENT)})
    s1 = create_node(node_id=2, properties={'group': get_quoted_string(SET)})

    e2 = create_node(node_id=3, properties={'group': get_quoted_string(ELEMENT)})
    s2 = create_node(node_id=4, properties={'group': get_quoted_string(SET)})

    e3 = create_node(node_id=5, properties={'group': get_quoted_string(ELEMENT)})
    s3 = create_node(node_id=6, properties={'group': get_quoted_string(SET)})

    insert_nodes(db, [e1, e2, e3, s1, s2, s3])
    insert_relationship(db, e1, s1)
    insert_relationship(db, e2, s2)
    insert_relationship(db, e3, s3)

    resulting_sets = set_cover(db, 'greedy')
    assert resulting_sets is not None


# Test suite 3
def test_greedy_result_number_of_sets_disjoint(db, db_delete):
    e1 = create_node(node_id=1, properties={'group': get_quoted_string(ELEMENT)})
    s1 = create_node(node_id=2, properties={'group': get_quoted_string(SET)})

    e2 = create_node(node_id=3, properties={'group': get_quoted_string(ELEMENT)})
    s2 = create_node(node_id=4, properties={'group': get_quoted_string(SET)})

    e3 = create_node(node_id=5, properties={'group': get_quoted_string(ELEMENT)})
    s3 = create_node(node_id=6, properties={'group': get_quoted_string(SET)})

    insert_nodes(db, [e1, e2, e3, s1, s2, s3])
    insert_relationship(db, e1, s1)
    insert_relationship(db, e2, s2)
    insert_relationship(db, e3, s3)

    resulting_sets = set_cover(db, 'greedy')
    assert len(resulting_sets) == len(set(x.id for x in chain(resulting_sets)))


# Test suite 4
def test_result_picked_optimized_solution(db, db_delete):
    e1 = create_node(node_id=1, properties={'group': get_quoted_string(ELEMENT)})
    s1 = create_node(node_id=2, properties={'group': get_quoted_string(SET)})

    e2 = create_node(node_id=3, properties={'group': get_quoted_string(ELEMENT)})
    s2 = create_node(node_id=4, properties={'group': get_quoted_string(SET)})

    e3 = create_node(node_id=5, properties={'group': get_quoted_string(ELEMENT)})
    s3 = create_node(node_id=6, properties={'group': get_quoted_string(SET)})

    insert_nodes(db, [e1, e2, e3, s1, s2, s3])
    insert_relationship(db, e1, s1)
    insert_relationship(db, e2, s1)
    insert_relationship(db, e2, s2)
    insert_relationship(db, e3, s3)

    resulting_sets = set_cover(db)
    assert len(resulting_sets) == 2