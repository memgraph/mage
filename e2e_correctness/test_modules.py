from contextlib import closing
import itertools
import socket
from time import sleep
from typing import Dict, List
import pytest
import yaml

from pathlib import Path
from gqlalchemy import Memgraph, Node

import logging
import neo4j
from docker_handler import remove_container, start_memgraph_mage_container, start_neo4j_apoc_container, stop_container
from query_neo_mem import Graph, clean_memgraph_db, clean_neo4j_db, create_memgraph_db, create_neo4j_driver, mg_execute_cyphers, mg_get_graph, neo4j_execute_cyphers, neo4j_get_graph, run_memgraph_query, run_neo4j_query

logging.basicConfig(format='%(asctime)-15s [%(levelname)s]: %(message)s')
logger = logging.getLogger('play_handler')
logger.setLevel(logging.INFO)



class TestConstants:
    ABSOLUTE_TOLERANCE = 1e-3

    EXCEPTION = "exception"
    INPUT_FILE = "input.cyp"
    OUTPUT = "output"
    QUERY = "query"
    TEST_MODULE_DIR_SUFFIX = "_test"
    TEST_GROUP_DIR_SUFFIX = "_group"

    ONLINE_TEST_E2E_SETUP = "setup"
    ONLINE_TEST_E2E_CLEANUP = "cleanup"
    ONLINE_TEST_E2E_INPUT_QUERIES = "queries"
    TEST_SUBDIR_PREFIX = "test"
    TEST_FILE = "test.yml"
    MEMGRAPH_QUERY = "memgraph_query"
    NEO4J_QUERY = "neo4j_query"

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
            TestConstants.TEST_MODULE_DIR_SUFFIX
        ):
            continue

        for test_or_group_dir in module_test_dir.iterdir():
            if not test_or_group_dir.is_dir():
                continue

            if test_or_group_dir.name.endswith(TestConstants.TEST_GROUP_DIR_SUFFIX):
                for test_dir in test_or_group_dir.iterdir():
                    if not test_dir.is_dir():
                        continue

                    tests.append(
                    #    pytest.param
                        
                            test_dir, 
                            #f"{module_test_dir.stem}-{test_or_group_dir.stem}-{test_dir.stem}",
                        
                    )
            else:
                tests.append(
                #    pytest.param(
                    
                        test_or_group_dir,
                        #f"{module_test_dir.stem}-{test_or_group_dir.stem}",
                    
                )
    return tests


tests = prepare_tests()


def _load_yaml(path: Path) -> Dict:
    """
    Load YAML based file in Python dictionary.
    """
    file_handle = path.open("r")
    return yaml.load(file_handle, Loader=yaml.Loader)


def _graphs_equal(memgraph_graph:Graph, neo4j_graph: Graph) -> bool:
    assert len(memgraph_graph.vertices) == len(neo4j_graph.vertices), f"Num of vertices is not same. \
        Memgraph contains {memgraph_graph.vertices} and Neo4j contains {neo4j_graph.vertices}"
    
    assert len(memgraph_graph.edges) == len(neo4j_graph.edges), f"Num of edges is not same. \
        Memgraph contains {memgraph_graph.edges} and Neo4j contains {neo4j_graph.edges}"

    for i, mem_vertex in enumerate(memgraph_graph.vertices):
        neo_vertex = neo4j_graph.vertices[i]
        assert mem_vertex == neo_vertex, f"Vertices are different.\
            Neo4j vertex: {neo_vertex}\
            Memgraph vertex: {mem_vertex}"
    for i, mem_edge in enumerate(memgraph_graph.edges):
        neo_edge = neo4j_graph.edges[i]
        assert neo_edge == mem_edge, f"Edges are different.\
            Neo4j edge: {neo_edge}\
            Memgraph edge: {mem_edge}"  
    
def _run_test(test_dir:str, memgraph_db: Memgraph, neo4j_driver: neo4j.BoltDriver):
    """
    Run input queries on Memgraph and Neo4j and compare graphs after running test query
    """
    input_cyphers = test_dir.joinpath(TestConstants.INPUT_FILE).open("r").readlines()
    mg_execute_cyphers(input_cyphers, memgraph_db)
    logger.info(f"Imported data into Memgraph from {input_cyphers}")
    neo4j_execute_cyphers(input_cyphers, neo4j_driver)
    logger.info(f"Imported data into Neo4j from {input_cyphers}")
    sleep(2)
    

    test_dict = _load_yaml(test_dir.joinpath(TestConstants.TEST_FILE))
    logger.info(f"Test dict {test_dict}")

    logger.info(f"Running query against Memgraph: {test_dict[TestConstants.MEMGRAPH_QUERY]}")
    run_memgraph_query(test_dict[TestConstants.MEMGRAPH_QUERY], memgraph_db)
    logger.info("Done")

    logger.info(f"Running query against Neo4j: {test_dict[TestConstants.NEO4J_QUERY]}")
    run_neo4j_query(test_dict[TestConstants.NEO4J_QUERY], neo4j_driver)
    logger.info("Done")

    mg_graph = mg_get_graph(memgraph_db)
    neo4j_graph = neo4j_get_graph(neo4j_driver)


    

    assert _graphs_equal(mg_graph, neo4j_graph)
       




def test_end2end(test_dir: Path, memgraph_db: Memgraph, neo4j_driver: neo4j.BoltDriver):
    clean_memgraph_db(memgraph_db)
    clean_neo4j_db(neo4j_driver)


    if test_dir.name.startswith(TestConstants.TEST_SUBDIR_PREFIX):
        _run_test(test_dir, memgraph_db, neo4j_driver)
    else:
        logger.info(f"Skipping following dir: {test_dir.name}")

    # Clean database once testing module is finished
    clean_memgraph_db(memgraph_db)
    clean_neo4j_db(neo4j_driver)

def cleanup_container(container_name:str, container_id:str) -> None:
    logger.info(f"Stopping {container_name} container with id: {container_id}")
    stop_container(container_id)
    logger.info(f"Stopped {container_name} container with id: {container_id}")


#@pytest.mark.parametrize("test_dir", tests)
#def test(tests):
def main():
    neo4j_port = 7688
    memgraph_port = 7687
    neo4j_container_name = "test_neo4j_apoc"
    memgraph_mage_container_name = "test_memgraph_mage"

    neo4j_image_name = "neo4j:latest"
    memgraph_mage_image_name = "memgraph-mage:test"

    neo4j_container_id = None
    memgraph_container_id = None
    try:
        
        neo4j_container_id = start_neo4j_apoc_container(image_name=neo4j_image_name, 
                                                    port=neo4j_port, 
                                                    container_name=neo4j_container_name)
    except Exception as e:
        print(e)
        if neo4j_container_id is not None:
            cleanup_container(neo4j_container_id)

    try:
        memgraph_container_id = start_memgraph_mage_container(image_name=memgraph_mage_image_name,
                                                          container_name=memgraph_mage_container_name,
                                                          port=memgraph_port)
    except Exception as e:
        print(e)
        if memgraph_container_id is not None:
            cleanup_container(memgraph_container_id)

    logger.info(f"Neo4j container id: {neo4j_container_id}")

    logger.info(f"Memgraph container id: {memgraph_container_id}")

    
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex(("localhost", 7687)) == 0:
            print("Port is open")
        else:
            print("Port is not open")
    sleep(10)
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex(("localhost", 7687)) == 0:
            print("Port is open")
        else:
            print("Port is not open")
    try: 

        neo4j_driver = create_neo4j_driver(neo4j_port)
        logger.info("Created neo4j driver")
        memgraph_db = create_memgraph_db(memgraph_port)
        logger.info("Created Memgraph connection")

        for test in tests:
            print(test)
            logger.info(f"Testing {test}")
            test_end2end(test, memgraph_db, neo4j_driver)
            logger.info(f"Done testing {test}")
    except Exception as e:
        print(e)
    finally:
        cleanup_container(neo4j_container_name, neo4j_container_id)
        cleanup_container(memgraph_mage_container_name, memgraph_container_id)
 

if __name__ == "__main__":
    main()