import pytest
from database import Memgraph, Node
from query_module import delete_all, create_node, insert_nodes, calculate_distance_multiple


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
def test_result_is_present_has_2_entries(db, db_delete):
    n11 = create_node(node_id=1, properties={'lat': 1, 'lng': 1}, group='a')
    n12 = create_node(node_id=2, properties={'lat': 1, 'lng': 2}, group='a')
    n21 = create_node(node_id=3, properties={'lat': 2, 'lng': 1}, group='b')
    n22 = create_node(node_id=4, properties={'lat': 2, 'lng': 2}, group='b')

    insert_nodes(db, [n11, n12, n21, n22])

    distances = calculate_distance_multiple(db, [n11, n12], [n21, n22])
    assert distances is not None
    assert len(distances) == 2


# Test suite 1
def test_result_is_correct(db, db_delete):
    n11 = create_node(node_id=1, properties={'lat': 1, 'lng': 1}, group='a')
    n12 = create_node(node_id=2, properties={'lat': 1, 'lng': 2}, group='a')
    n21 = create_node(node_id=3, properties={'lat': 2, 'lng': 1}, group='b')
    n22 = create_node(node_id=4, properties={'lat': 2, 'lng': 2}, group='b')

    insert_nodes(db, [n11, n12, n21, n22])

    distance = calculate_distance_multiple(db, [n11, n12], [n21, n22])
    assert distance[0] > 111100 and distance[0] < 111300
    assert distance[1] > 111100 and distance[1] < 111300


# Test suite 2
def test_result_is_correct_in_metres(db, db_delete):
    n11 = create_node(node_id=1, properties={'lat': 1, 'lng': 1}, group='a')
    n12 = create_node(node_id=2, properties={'lat': 1, 'lng': 2}, group='a')
    n21 = create_node(node_id=3, properties={'lat': 2, 'lng': 1}, group='b')
    n22 = create_node(node_id=4, properties={'lat': 2, 'lng': 2}, group='b')

    insert_nodes(db, [n11, n12, n21, n22])

    distance = calculate_distance_multiple(db, [n11, n12], [n21, n22], 'm')
    assert distance[0] > 111100 and distance[0] < 111300
    assert distance[1] > 111100 and distance[1] < 111300


# Test suite 3
def test_result_is_correct_in_kilometres(db, db_delete):
    n11 = create_node(node_id=1, properties={'lat': 1, 'lng': 1}, group='a')
    n12 = create_node(node_id=2, properties={'lat': 1, 'lng': 2}, group='a')
    n21 = create_node(node_id=3, properties={'lat': 2, 'lng': 1}, group='b')
    n22 = create_node(node_id=4, properties={'lat': 2, 'lng': 2}, group='b')

    insert_nodes(db, [n11, n12, n21, n22])

    distance = calculate_distance_multiple(db, [n11, n12], [n21, n22], metrics='km')
    assert distance[0] > 111.100 and distance[0] < 111.300
    assert distance[1] > 111.100 and distance[1] < 111.300


# Test suite 4
def test_result_is_incorrect_lat_not_inserted(db, db_delete):
    with pytest.raises(Exception):
        n11 = create_node(node_id=1, properties={'lat': 1, 'lng': 1}, group='a')
        n12 = create_node(node_id=2, properties={'lng': 2}, group='a')
        n21 = create_node(node_id=3, properties={'lat': 2, 'lng': 1}, group='b')
        n22 = create_node(node_id=4, properties={'lat': 2, 'lng': 2}, group='b')

        insert_nodes(db, [n11, n12, n21, n22])

        calculate_distance_multiple(db, [n11, n12], [n21, n22], metrics='km')


# Test suite 5
def test_result_is_incorrect_lng_not_inserted(db, db_delete):
    with pytest.raises(Exception):
        n11 = create_node(node_id=1, properties={'lat': 1, 'lng': 1}, group='a')
        n12 = create_node(node_id=2, properties={'lng': 2}, group='a')
        n21 = create_node(node_id=3, properties={'lat': 2, 'lng': 1}, group='b')
        n22 = create_node(node_id=4, properties={'lat': 2}, group='b')

        insert_nodes(db, [n11, n12, n21, n22])

        calculate_distance_multiple(db, [n11, n12], [n21, n22], metrics='km')
