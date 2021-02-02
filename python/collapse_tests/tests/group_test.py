import pytest
from database import Memgraph
from query_module import delete_all, make_graph, groups


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


# Test suite 1
def test_groups_one_level_hierarchy(db, db_delete):
    graph = {
        "c:C": ["p1:P:Edge", "p2:P:Edge"],
        "p1:P": ["s1:S:Edge", "s2:S:Edge"],
        "p2:P": ["s3:S:Edge", "s4:S:Edge"],
        "s3:S": ["s2:S:Friend"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S', 's2:S']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P', 's3:S', 's4:S']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge"])
    assert compare(result, expected)


# Test suite 2
def test_groups_pseudo_label(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "p2:P": ["s2:S:Edge"],
        "t:T": ["s2:S:Transport"],
        "s1:S": ["t:T:Transport"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P', 's2:S']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge"], pseudo_labels=['T'])
    assert compare(result, expected)


# Test suite 3
def test_groups_direct_nodes_same_level(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge", "p2:P:Connected"],
        "p2:P": ["s2:S:Edge"],
        "s2:S": ["s1:S:Friend"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P', 's2:S']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge"])
    assert compare(result, expected)


# Test suite 4
def test_groups_reverse_direction(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "s2:S": ["p2:P:Edge"],
        "s1:S": ["s2:S:Friend"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge"])
    assert compare(result, expected)


# Test suite 5
def test_groups_two_level_hierarchy(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "p2:P": ["s2:S:Edge"],
        "s1:S": ["n1:N:Next", "n2:N:Next"],
        "s2:S": ["n3:N:Next"],
        "n3:N": ["n1:N:Friend"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S', 'n1:N', 'n2:N']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P', 's2:S', 'n3:N']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge", "Next"])
    assert compare(result, expected)


# Test suite 6
def test_groups_two_level_hierarchy_node_sharing(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "p2:P": ["s2:S:Edge"],
        "s1:S": ["n1:N:Next"],
        "s2:S": ["n1:N:Next"],
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P', 's2:S']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge"])
    assert compare(result, expected)


# Test suite 7
def test_groups_common_connection(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "p2:P": ["s1:S:Edge"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P']}
    ]
    make_graph(db, graph)
    with pytest.raises(Exception):
        groups(db, vertex="P", edge_types=["Edge", "Transport"])


# Test suite 8
def test_groups_multiple_label_direct_mix(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge", "t:T:Transport"],
        "p2:P": ["s2:S:Edge"],
        "s1:S": ["s2:S:Friend"],
        "t:T": ["p3:P:Transport"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S', 't:T']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P', 's2:S']},
        {'top_vertex': 'p3:P', 'collapsed_vertices': ['p3:P']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge", "Transport"])
    assert compare(result, expected)


# Test suite 9
def test_groups_pseudo_label_mix(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "p2:P": ["s2:S:Edge"],
        "p3:P": ["s3:S:Edge"],
        "s1:S": ["s2:S:Friend", "t:T:Transport"],
        "t:T": ["s3:S:Transport"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P', 's2:S']},
        {'top_vertex': 'p3:P', 'collapsed_vertices': ['p3:P', 's3:S']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge"], pseudo_labels=["T"])
    assert compare(result, expected)


def test_groups_pseudo_label_additional_argument(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "p2:P": ["s2:S:Edge"],
        "p3:P": ["s3:S:Edge"],
        "s1:S": ["s2:S:Friend", "t:T:Transport"],
        "t:T": ["s3:S:Transport"]
    }

    make_graph(db, graph)
    with pytest.raises(Exception):
        groups(db, vertex="P", edge_types=["Edge", "Transport"], pseudo_labels=["T"])


# Test suite 10
def test_groups_single_connection(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "s1:S": ["p1:P:Edge"],
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge"])
    assert compare(result, expected)


# Test suite 11
def test_groups_multiple_label_mix(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge", "t1:T:Transport"],
        "p2:P": ["s2:S:Edge"],
        "p3:P": ["t3:T:Transport"],
        "s1:S": ["s2:S:Friend"],
        "t3:T": ["t1:T:Next"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S', 't1:T']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P', 's2:S']},
        {'top_vertex': 'p3:P', 'collapsed_vertices': ['p3:P', 't3:T']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge", "Transport"])
    assert compare(result, expected)


# Test suite 12
def test_groups_bidirectional(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "p2:P": ["s2:S:Edge"],
        "s1:S": ["s2:S:Friend"],
        "s2:S": ["s1:S:Friend"],
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P', 's2:S']},
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge"])
    assert compare(result, expected)


# Test suite 13
def test_groups_bidirectional_pseudo_label(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "p2:P": ["s2:S:Edge"],
        "s1:S": ["t:T:Transport"],
        "s2:S": ["t:T:Transport"],
        "t:T": ["s1:S:Transport", "s2:S:Transport"],
    }

    make_graph(db, graph)
    with pytest.raises(Exception):
        groups(db, vertex="P", edge_types=["Edge"], pseudo_labels=["T"])


# Test suite 14
def test_groups_different_labels(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge"],
        "p2:P": ["s2:S:Edge"],
        "s2:S": ["n2:N:Next"],
        "n2:N": ["s1:S:Friend"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S']},
        {'top_vertex': 'p2:P', 'collapsed_vertices': ['p2:P', 's2:S', 'n2:N']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge", "Next"])
    assert compare(result, expected)


# Test suite 15
def test_groups_single_recursive_node(db, db_delete):
    graph = {
        "p1:P": ["p1:P:Edge"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge"])
    assert compare(result, expected)


# Test suite 16
def test_groups_single_node_hierarchy(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge", "s2:S:Edge"],
        "s2:S": ["s1:S:Friend"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S', 's2:S']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge"])
    assert compare(result, expected)


# Test suite 17
def test_groups_hierarchy_recursive(db, db_delete):
    graph = {
        "p1:P": ["s1:S:Edge", "s2:S:Edge"],
        "s1:S": ["n1:N:Next"],
        "n1:N": ["p1:P:Connect"]
    }

    expected = [
        {'top_vertex': 'p1:P', 'collapsed_vertices': ['p1:P', 's1:S', 's2:S', 'n1:N']}
    ]
    make_graph(db, graph)
    result = groups(db, vertex="P", edge_types=["Edge", "Next"])
    assert compare(result, expected)
