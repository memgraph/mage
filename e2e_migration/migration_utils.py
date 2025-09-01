"""
Migration testing utilities for e2e_migration tests.
Provides functions to connect to various databases and Memgraph, execute queries, and compare results.
"""
import logging
import pytest
import yaml
import os
from typing import Dict, List, Any, Optional
from gqlalchemy import Memgraph

logging.basicConfig(format="%(asctime)-15s [%(levelname)s]: %(message)s")
logger = logging.getLogger("e2e_migration")
logger.setLevel(logging.INFO)


class MigrationTestConstants:
    """Constants for migration testing."""
    EXCEPTION = "exception"
    INPUT_FILE = "input.cyp"
    OUTPUT = "output"
    QUERY = "query"
    TEST_MODULE_DIR_SUFFIX = "_test"
    MIGRATION_QUERY = "migration_query"
    EXPECTED_RESULT = "expected_result"
    SOURCE_QUERY = "source_query"





class MemgraphConnection:
    """Memgraph database connection wrapper."""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connection = None
    
    def connect(self):
        """Establish connection to Memgraph database."""
        try:
            self.connection = Memgraph(host=self.host, port=self.port)
            logger.info(f"Connected to Memgraph at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Memgraph: {e}")
            raise
    
    def disconnect(self):
        """Close Memgraph connection."""
        if self.connection:
            self.connection.close()
        logger.info("Disconnected from Memgraph")
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        if not self.connection:
            raise RuntimeError("Not connected to Memgraph")
        
        try:
            results = list(self.connection.execute_and_fetch(query))
            logger.info(f"Executed Memgraph query, returned {len(results)} rows")
            return results
        except Exception as e:
            logger.error(f"Failed to execute Memgraph query: {e}")
            raise
    
    def clean_database(self):
        """Remove all nodes and relationships from Memgraph."""
        try:
            self.connection.execute("MATCH (n) DETACH DELETE n")
            logger.info("Cleaned Memgraph database")
        except Exception as e:
            logger.error(f"Failed to clean Memgraph database: {e}")
            raise


def load_test_config(test_file_path: str) -> Dict[str, Any]:
    """Load test configuration from YAML file."""
    try:
        with open(test_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Failed to load test config from {test_file_path}: {e}")
        raise


def setup_memgraph_connection() -> MemgraphConnection:
    """Setup Memgraph connection with fixed localhost:7687."""
    memgraph_conn = MemgraphConnection(host="localhost", port=7687)
    memgraph_conn.connect()
    return memgraph_conn
