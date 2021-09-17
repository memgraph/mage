from typing import Dict, List
import pytest
import yaml

from pathlib import Path
from database import Memgraph, Node


@pytest.fixture
def db():
    return Memgraph()


class TestConstants:
    ABSOLUTE_TOLERANCE = 1e-3

    CLEANUP = "cleanup"
    EXCEPTION = "exception"
    INPUT_FILE = "input.cyp"
    OUTPUT = "output"
    ONLINE_TEST = "_test"
    SETUP = "setup"
    QUERY = "query"
    QUERIES = "queries"
    TEST_DIR_ENDING = "_test"
    TEST_DIR_ONLINE_PREFIX = "test_online"
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
    """
    Fetch all the tests in the testing folders, and prepare them for execution
    """
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


def _load_yaml(path: Path) -> Dict:
    """
    Load YAML based file in Python dictionary.
    """
    file_handle = path.open("r")
    return yaml.load(file_handle, Loader=yaml.Loader)


def _execute_cyphers(input_cyphers: List[str], db: Memgraph):
    """
    Execute commands against Memgraph
    """
    for query in input_cyphers:
        db.execute(query)


def _run_test(test_dict: Dict, db: Memgraph):
    """
    Run queries on Memgraph and compare them to expected results stored in test_dict
    """
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


def _test_static(test_dir: Path, db: Memgraph):
    """
    Testing static modules.
    """
    input_cyphers = test_dir.joinpath(TestConstants.INPUT_FILE).open("r").readlines()
    _execute_cyphers(input_cyphers, db)

    test_dict = _load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))
    _run_test(test_dict, db)


def _test_online(test_dir: Path, db: Memgraph):
    """
    Testing online modules. Checkpoint testing
    """
    checkpoint_input = _load_yaml(test_dir.joinpath(TestConstants.INPUT_FILE))
    checkpoint_test_dicts = _load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))

    setup_cyphers = checkpoint_input[TestConstants.SETUP]
    checkpoint_input_cyphers = checkpoint_input[TestConstants.QUERIES]
    cleanup_cyphers = checkpoint_input[TestConstants.CLEANUP]

    _execute_cyphers(setup_cyphers.splitlines(), db)
    for input_cyphers_raw, test_dict in zip(
        checkpoint_input_cyphers, checkpoint_test_dicts
    ):
        input_cyphers = input_cyphers_raw.splitlines()
        _execute_cyphers(input_cyphers, db)
        _run_test(test_dict, db)
    _execute_cyphers(cleanup_cyphers.splitlines(), db)


@pytest.mark.parametrize("test_dir", tests)
def test_end2end(test_dir: Path, db: Memgraph):
    db.drop_database()

    if not test_dir.name.startswith(TestConstants.TEST_DIR_ONLINE_PREFIX):
        _test_static(test_dir, db)
    else:
        _test_online(test_dir, db)

    # Clean database once testing module is finished
    db.drop_database()
