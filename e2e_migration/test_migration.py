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
        
        # Load test configuration
        test_config_path = f"{test_dir}/test/test_migration.yml"
        test_config = load_test_config(test_config_path)
        
        # Execute migration query
        migration_query = test_config["query"]
        memgraph_results = self.memgraph.execute_query(migration_query)
        
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
