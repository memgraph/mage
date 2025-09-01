def pytest_addoption(parser):
    parser.addoption("--test-dir", type=str, action="store", help="Test directory name")
