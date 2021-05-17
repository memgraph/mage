import pytest
import yaml

from pathlib import Path
from database import Memgraph, Node


@pytest.fixture
def db():
    return Memgraph()


class TestConstants:
    ABSOLUTE_TOLERANCE = 1e-3

    EXCEPTION = "exception"
    INPUT_FILE = "input.cyp"
    OUTPUT = "output"
    QUERY = "query"
    TEST_DIR_ENDING = "_test"
    TEST_FILE = "test.yml"


def _node_to_dict(data):
    labels = data.labels if hasattr(data, "labels") else data._labels
    properties = data.properties if hasattr(data, "properties") else data._properties
    return {"labels": list(labels), "properties": properties}


def _replace(data, match_classes):
    if isinstance(data, dict):
        return {k: _replace(v, match_classes) for k, v in data.items()}
    elif isinstance(data, list):
        return [_replace(i, match_classes) for i in data]
    elif isinstance(data, float):
        return pytest.approx(data, abs=TestConstants.ABSOLUTE_TOLERANCE)
    else:
        return _node_to_dict(data) if isinstance(data, match_classes) else data


def prepare_tests():
    tests = []

    test_path = Path().cwd()
    for module_test_dir in test_path.iterdir():
        if not module_test_dir.is_dir() or not module_test_dir.name.endswith(
            TestConstants.TEST_DIR_ENDING
        ):
            continue

        for test_dir in module_test_dir.iterdir():
            if not test_dir.is_dir():
                continue
            tests.append(
                pytest.param(test_dir, id=f"{module_test_dir.stem}-{test_dir.stem}")
            )
    return tests


tests = prepare_tests()


@pytest.mark.parametrize("test_dir", tests)
def test_end2end(test_dir, db):
    db.drop_database()

    input_cyphers = test_dir.joinpath(TestConstants.INPUT_FILE).open("r").readlines()
    for query in input_cyphers:
        db.execute_query(query)

    test_dict = yaml.load(
        test_dir.joinpath(TestConstants.TEST_FILE).open("r"), Loader=yaml.Loader
    )

    test_query = test_dict[TestConstants.QUERY]

    output_test = TestConstants.OUTPUT in test_dict
    exception_test = TestConstants.EXCEPTION in test_dict

    if not (output_test ^ exception_test):
        pytest.fail("Test file has no valid format.")

    if output_test:
        result_query = list(db.execute_and_fetch(test_query))
        result = _replace(result_query, Node)

        expected = test_dict[TestConstants.OUTPUT]
        assert result == expected

    if exception_test:
        # TODO: Implement for different kinds of errors
        try:
            result = db.execute_and_fetch(test_query)
            assert result is None
        except Exception:
            assert True
