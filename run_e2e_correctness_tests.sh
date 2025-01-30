#!/bin/bash -e

MEMGRAPH_PORT=$1
NEO4J_PORT=$2
NEO4J_CONTAINER=$3

timeout=30 
counter=0

docker run --rm \
          --name "$NEO4J_CONTAINER"  \
          -p 7474:7474 -p "$NEO4J_PORT":7687 \
          -d \
          -v "$HOME/neo4j/plugins:/plugins" \
          --env NEO4J_AUTH=none  \
          -e NEO4J_apoc_export_file_enabled=true \
          -e NEO4J_apoc_import_file_enabled=true \
          -e NEO4J_apoc_import_file_use__neo4j__config=true  \
          -e NEO4J_PLUGINS='["apoc"]' neo4j:5.10.0

echo "Waiting for Neo4j to start..."
start_time=$(date +%s)
while ! curl --silent --fail http://localhost:7474; do
  sleep 1
    counter=$((counter+1))
  if [ $counter -gt $timeout ]; then
    echo "Neo4j failed to start in $timeout seconds"
    exit 1
  fi
done

echo "Neo4j started. Running tests..."
python3 test_e2e_correctness.py --memgraph-port "$MEMGRAPH_PORT" --neo4j-port "$NEO4J_PORT"

echo "Stopping Neo4j..."
docker stop "$NEO4J_CONTAINER"