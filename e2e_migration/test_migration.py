"""
Migration testing module for e2e_migration tests.
Tests the migration of data from various databases to Memgraph using the migrate module.
"""
import logging
import pytest

from migration_utils import (
    load_test_config,
    setup_memgraph_connection
)


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption("--test-dir", action="store", help="Test directory name")
    parser.addoption("--test-file", action="store", help="Test file path relative to test directory")

logging.basicConfig(format="%(asctime)-15s [%(levelname)s]: %(message)s")
logger = logging.getLogger("e2e_migration")
logger.setLevel(logging.INFO)


class TestMigration:
    """Test class for migration functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test."""
        # Setup Memgraph connection only
        self.memgraph = setup_memgraph_connection()
        
        # Clean Memgraph before each test
        self.memgraph.clean_database()
        
        yield
        
        # Cleanup after each test
        self.memgraph.clean_database()
        self.memgraph.disconnect()
    
    def test_data_migration(self, request):
        """Test migration of data from database to Memgraph."""
        # Get test directory from pytest parameter
        test_dir = request.config.getoption("--test-dir")
        if not test_dir:
            raise RuntimeError("--test-dir parameter is required")
        
        # Get test file from pytest parameter
        test_file = request.config.getoption("--test-file")
        if not test_file:
            raise RuntimeError("--test-file parameter is required")
        
        # Load test configuration
        test_config_path = f"{test_dir}/{test_file}"
        test_config = load_test_config(test_config_path)
        
        # Check if this test expects an exception
        expect_exception = test_config.get("exception", False)
        
        try:
            # Execute migration query
            migration_query = test_config["query"]
            memgraph_results = self.memgraph.execute_query(migration_query)
            
            if expect_exception:
                # If we expected an exception but got results, the test should fail
                pytest.fail("Expected migration to fail with an exception, but it succeeded")
            
            # Get expected results from test configuration
            expected_results = test_config["output"]
            
            # Compare results
            assert len(memgraph_results) == len(expected_results), f"Result count mismatch: expected {len(expected_results)}, got {len(memgraph_results)}"
            
            # Validate data by comparing with expected output
            for i, (actual_row, expected_row) in enumerate(zip(memgraph_results, expected_results)):
                for field, expected_value in expected_row.items():
                    actual_value = actual_row.get(field)
                    
                    # Simple equality comparison for all data types
                    assert actual_value == expected_value, f"Field {field} mismatch at row {i}: expected {expected_value}, got {actual_value}"
            
            logger.info("Data migration to Memgraph successfully validated!")
            
        except Exception as e:
            if expect_exception:
                # Expected exception - test passes
                logger.info(f"Migration failed as expected: {e}")
                return
            else:
                # Unexpected failure - re-raise the exception
                raise
