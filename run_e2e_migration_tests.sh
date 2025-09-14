#!/bin/bash -e

# Container names and images (set via command line arguments)
MAGE_CONTAINER=""
MYSQL_CONTAINER=""
POSTGRESQL_CONTAINER=""
NEO4J_CONTAINER=""

# Parse command line arguments
TEST_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -k)
            TEST_FILTER="$2"
            shift 2
            ;;
        --mage-container)
            MAGE_CONTAINER="$2"
            shift 2
            ;;
        --mysql-container)
            MYSQL_CONTAINER="$2"
            shift 2
            ;;
        --postgresql-container)
            POSTGRESQL_CONTAINER="$2"
            shift 2
            ;;
        --neo4j-container)
            NEO4J_CONTAINER="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [-k FILTER] --mage-container CONTAINER [--mysql-container CONTAINER] [--postgresql-container CONTAINER] [--neo4j-container CONTAINER] [--mysql-image IMAGE] [--postgresql-image IMAGE]"
            echo "  -k FILTER                    Filter tests by database type (e.g., 'mysql', 'postgresql', 'neo4j')"
            echo "  --mage-container NAME        MAGE container name (required)"
            echo "  --mysql-container NAME       MySQL container name (required if mysql tests are run)"
            echo "  --postgresql-container NAME  PostgreSQL container name (required if postgresql tests are run)"
            echo "  --neo4j-container NAME       Neo4j container name (required if neo4j tests are run)"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required arguments based on test filter
if [ -z "$MAGE_CONTAINER" ]; then
    echo "Error: MAGE container name is required"
    echo "Usage: $0 --mage-container CONTAINER [other options]"
    exit 1
fi

# Check if MySQL tests will be run
if [ -z "$TEST_FILTER" ] || [ "$TEST_FILTER" = "mysql" ]; then
    if [ -z "$MYSQL_CONTAINER" ]; then
        echo "Error: MySQL container name and image are required for MySQL tests"
        echo "Usage: $0 --mage-container CONTAINER --mysql-container CONTAINER [other options]"
        exit 1
    fi
fi

# Check if PostgreSQL tests will be run
if [ -z "$TEST_FILTER" ] || [ "$TEST_FILTER" = "postgresql" ]; then
    if [ -z "$POSTGRESQL_CONTAINER" ]; then
        echo "Error: PostgreSQL container name and image are required for PostgreSQL tests"
        echo "Usage: $0 --mage-container CONTAINER --postgresql-container CONTAINER [other options]"
        exit 1
    fi
fi

# Check if Neo4j tests will be run
if [ -z "$TEST_FILTER" ] || [ "$TEST_FILTER" = "neo4j" ]; then
    if [ -z "$NEO4J_CONTAINER" ]; then
        echo "Error: Neo4j container name is required for Neo4j tests"
        echo "Usage: $0 --mage-container CONTAINER --neo4j-container CONTAINER [other options]"
        exit 1
    fi
fi

wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=30
    
    echo "Waiting for $service_name to start..."
    counter=0
    while [ $counter -lt $timeout ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "$service_name is up and running."
            return 0
        fi
        sleep 1
        counter=$((counter+1))
    done
    
    echo "$service_name failed to start in $timeout seconds"
    return 1
}

run_mysql_tests() {
    echo "Starting MySQL..."
    
    # Start MySQL using docker compose with inline environment variables
    cd e2e_migration/test_mysql
    MYSQL_CONTAINER="$MYSQL_CONTAINER" docker compose up -d
    sleep 30
    if ! wait_for_service "localhost" 3306 "MySQL"; then
        MYSQL_CONTAINER="$MYSQL_CONTAINER" docker compose down -v 2>/dev/null || true
        cd ../..
        return 1
    fi
    
    echo "Running MySQL migration tests..."
    cd ../..
    docker exec -i -u memgraph "$MAGE_CONTAINER" bash -c "cd /mage && python3 -m pytest e2e_migration/test_migration.py -v -k mysql"
    
    echo "Stopping MySQL..."
    cd e2e_migration/test_mysql
    MYSQL_CONTAINER="$MYSQL_CONTAINER" docker compose down -v
    cd ../..
}

run_postgresql_tests() {
    echo "Starting PostgreSQL..."
    
    # Start PostgreSQL using docker compose with inline environment variables
    cd e2e_migration/test_postgresql
    POSTGRESQL_CONTAINER="$POSTGRESQL_CONTAINER" docker compose up -d
    sleep 30
    if ! wait_for_service "localhost" 5432 "PostgreSQL"; then
        POSTGRESQL_CONTAINER="$POSTGRESQL_CONTAINER" docker compose down -v 2>/dev/null || true
        cd ../..
        return 1
    fi
    
    echo "Running PostgreSQL migration tests..."
    cd ../..
    docker exec -i -u memgraph "$MAGE_CONTAINER" bash -c "cd /mage && python3 -m pytest e2e_migration/test_migration.py -v -k postgresql"
    
    echo "Stopping PostgreSQL..."
    cd e2e_migration/test_postgresql
    POSTGRESQL_CONTAINER="$POSTGRESQL_CONTAINER" docker compose down -v
    cd ../..
}

run_neo4j_tests() {
    echo "Starting Neo4j..."
    
    # Start Neo4j using docker compose with inline environment variables
    cd e2e_migration/test_neo4j
    NEO4J_CONTAINER="$NEO4J_CONTAINER" docker compose up -d
    sleep 30
    if ! wait_for_service "localhost" 7474 "Neo4j"; then
        NEO4J_CONTAINER="$NEO4J_CONTAINER" docker compose down -v 2>/dev/null || true
        cd ../..
        return 1
    fi
    
    echo "Running Neo4j migration tests..."
    cd ../..
    docker exec -i -u memgraph "$MAGE_CONTAINER" bash -c "cd /mage && python3 -m pytest e2e_migration/test_migration.py -v -k neo4j"
    
    echo "Stopping Neo4j..."
    cd e2e_migration/test_neo4j
    NEO4J_CONTAINER="$NEO4J_CONTAINER" docker compose down -v
    cd ../..
}

# Main execution
if [ -z "$TEST_FILTER" ] || [ "$TEST_FILTER" = "mysql" ]; then
    run_mysql_tests
fi

if [ -z "$TEST_FILTER" ] || [ "$TEST_FILTER" = "postgresql" ]; then
    run_postgresql_tests
fi

if [ -z "$TEST_FILTER" ] || [ "$TEST_FILTER" = "neo4j" ]; then
    run_neo4j_tests
fi
