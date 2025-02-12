#!/bin/bash -e

MEMGRAPH_PORT=$1
NEO4J_PORT=$2
NEO4J_CONTAINER=$3
MAGE_CONTAINER=$4
MEMGRAPH_NETWORK=$5

echo "Start Neo4j..."
docker run --rm \
    --name "$NEO4J_CONTAINER"  \
    --network "$MEMGRAPH_NETWORK" \
    -d \
    -v "$HOME/neo4j/plugins:/plugins" \
    --env NEO4J_AUTH=none  \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true  \
    -e NEO4J_PLUGINS='["apoc"]' neo4j:5.10.0

echo "Installing python3 dependencies..."
docker exec -i -u root "$MAGE_CONTAINER" bash -c "pip install -r /mage/python/tests/requirements.txt --break-system-packages"

echo "Running e2e correctness tests..."
docker exec -i -u root "$MAGE_CONTAINER" bash -c "cd /mage/e2e_correctness && python3 test_e2e_correctness.py --memgraph-port $MEMGRAPH_PORT --neo4j-port $NEO4J_PORT --neo4j-container $NEO4J_CONTAINER"

echo "Stopping Neo4j..."
docker stop "$NEO4J_CONTAINER"
