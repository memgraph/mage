import pytest
from database import Memgraph, Node
from query_module import delete_all, create_node, insert_nodes, calculate_distance


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
    n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 2})
    n2 = create_node(node_id=2, properties={'lat': 2, 'lng': 3})

    insert_nodes(db, [n1, n2])

    distance = calculate_distance(db, n1, n2)
    assert distance is not None


# Test suite 2
def test_result_is_calculated_correctly(db, db_delete):
    n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 2})
    n2 = create_node(node_id=2, properties={'lat': 2, 'lng': 3})

    insert_nodes(db, [n1, n2])

    distance = calculate_distance(db, n1, n2)
    assert distance > 157200 and distance < 157250


# Test suite 3
def test_result_in_kilometres_correct(db, db_delete):
    n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 2})
    n2 = create_node(node_id=2, properties={'lat': 2, 'lng': 3})

    insert_nodes(db, [n1, n2])

    distance = calculate_distance(db, n1, n2, metrics='km')
    assert distance > 157.200 and distance < 157.250


# Test suite 4
def test_result_in_meters_correct(db, db_delete):
    n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 2})
    n2 = create_node(node_id=2, properties={'lat': 2, 'lng': 3})

    insert_nodes(db, [n1, n2])

    distance = calculate_distance(db, n1, n2, metrics='m')
    assert distance > 157200 and distance < 157250


# Test suite 5
def test_invalid_metrics_incorrect(db, db_delete):
    with pytest.raises(Exception):
        n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 2})
        n2 = create_node(node_id=2, properties={'lat': 2, 'lng': 3})

        insert_nodes(db, [n1, n2])

        calculate_distance(db, n1, n2, metrics='z')


# Test suite 6
def test_first_no_lat_incorrect(db, db_delete):
    with pytest.raises(Exception):
        n1 = create_node(node_id=1, properties={'lng': 2})
        n2 = create_node(node_id=2, properties={'lat': 2, 'lng': 3})

        insert_nodes(db, [n1, n2])

        calculate_distance(db, n1, n2)


# Test suite 7
def test_first_no_lng_incorrect(db, db_delete):
    with pytest.raises(Exception):
        n1 = create_node(node_id=1, properties={'lat': 1})
        n2 = create_node(node_id=2, properties={'lat': 2, 'lng': 3})

        insert_nodes(db, [n1, n2])

        calculate_distance(db, n1, n2)


# Test suite 8
def test_second_no_lat_incorrect(db, db_delete):
    with pytest.raises(Exception):
        n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 2})
        n2 = create_node(node_id=2, properties={'lng': 3})

        insert_nodes(db, [n1, n2])

        calculate_distance(db, n1, n2)


# Test suite 9
def test_second_no_lng_incorrect(db, db_delete):
    with pytest.raises(Exception):
        n1 = create_node(node_id=1, properties={'lat': 1, 'lng': 2})
        n2 = create_node(node_id=2, properties={'lat': 2})

        insert_nodes(db, [n1, n2])

        calculate_distance(db, n1, n2)