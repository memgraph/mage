

def pytest_addoption(parser):
    parser.addoption("--memgraph-port", type=int, action="store")
    parser.addoption("--neo4j-port", type=int, action="store")
    parser.addoption("--path-option", type=bool, action="store")
