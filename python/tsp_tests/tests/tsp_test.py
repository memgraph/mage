import pytest
from database import Memgraph, Node
from query_module import delete_all, create_node, insert_nodes, tsp


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
def test_result_is_present_has_3_entries(db, db_delete):
    n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 1})
    n2 = create_node(node_id=2, properties={'lat': 1, 'lng': 2})
    n3 = create_node(node_id=3, properties={'lat': 2, 'lng': 1})

    insert_nodes(db, [n1, n2, n3])

    sources, destinations = tsp(db)
    assert sources is not None
    assert destinations is not None
    assert len(sources) == 3
    assert len(destinations) == 3

# Test suite 2
def test_greedy_result_correct(db, db_delete):
    n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 1})
    n2 = create_node(node_id=2, properties={'lat': 1, 'lng': 2})
    n3 = create_node(node_id=3, properties={'lat': 2, 'lng': 1})

    insert_nodes(db, [n1, n2, n3])

    sources, destinations = tsp(db, 'greedy')
    assert sources is not None
    assert destinations is not None
    assert len(sources) == 3
    assert len(destinations) == 3


# Test suite 3
def test_2_approx_result_correct(db, db_delete):
    n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 1})
    n2 = create_node(node_id=2, properties={'lat': 1, 'lng': 2})
    n3 = create_node(node_id=3, properties={'lat': 2, 'lng': 1})

    insert_nodes(db, [n1, n2, n3])

    sources, destinations = tsp(db, '2_approx')
    assert sources is not None
    assert destinations is not None
    assert len(sources) == 3
    assert len(destinations) == 3

# Test suite 4
def test_1_5_approx_result_correct(db, db_delete):
    n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 1})
    n2 = create_node(node_id=2, properties={'lat': 1, 'lng': 2})
    n3 = create_node(node_id=3, properties={'lat': 2, 'lng': 1})

    insert_nodes(db, [n1, n2, n3])

    sources, destinations = tsp(db, '1.5_approx')
    assert sources is not None
    assert destinations is not None
    assert len(sources) == 3
    assert len(destinations) == 3


# Test suite 5
def test_first_source_is_same_as_last_dest(db, db_delete):
    n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 1})
    n2 = create_node(node_id=2, properties={'lat': 1, 'lng': 2})
    n3 = create_node(node_id=3, properties={'lat': 2, 'lng': 1})

    insert_nodes(db, [n1, n2, n3])

    sources, destinations = tsp(db)
    assert sources[0] == destinations[-1]


# Test suite 6
def test_transitive_source_destination_mapping(db, db_delete):
    n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 1})
    n2 = create_node(node_id=2, properties={'lat': 1, 'lng': 2})
    n3 = create_node(node_id=3, properties={'lat': 2, 'lng': 1})
    n4 = create_node(node_id=4, properties={'lat': 3, 'lng': 1})
    n5 = create_node(node_id=5, properties={'lat': 2, 'lng': 4})


    insert_nodes(db, [n1, n2, n3, n4, n5])

    sources, destinations = tsp(db)

    for idx in range(len(sources)-1):
        assert destinations[idx] == sources[idx + 1]


# Test suite 7
def test_one_lat_not_present_error(db, db_delete):
    with pytest.raises(Exception):
        n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 1})
        n2 = create_node(node_id=2, properties={'lat': 1, 'lng': 2})
        n3 = create_node(node_id=3, properties={'lng': 1})

        insert_nodes(db, [n1, n2, n3])
        tsp(db)


# Test suite 8
def test_invalid_solving_method(db, db_delete):
    with pytest.raises(Exception):
        n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 1})
        n2 = create_node(node_id=2, properties={'lat': 1, 'lng': 2})
        n3 = create_node(node_id=3, properties={'lng': 1})

        insert_nodes(db, [n1, n2, n3])
        tsp(db, 'some_method')