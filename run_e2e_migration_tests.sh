#!/bin/bash -e

# Default MAGE container name
MAGE_CONTAINER=${MAGE_CONTAINER:-"mage"}

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
        --help)
            echo "Usage: $0 [-k FILTER] [--mage-container CONTAINER]"
            echo "  -k FILTER              Filter tests by database type (e.g., 'mysql', 'postgresql')"
            echo "  --mage-container NAME  MAGE container name (default: mage)"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done


wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
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
    
    # Start MySQL using docker compose
    cd e2e_migration/test_mysql
    docker compose up -d
    
    if ! wait_for_service "localhost" 3306 "MySQL"; then
        docker compose down -v 2>/dev/null || true
        cd ../..
        return 1
    fi
    
    echo "Running MySQL migration tests..."
    cd ../..
    docker exec -i -u memgraph "$MAGE_CONTAINER" bash -c "cd /mage && python3 -m pytest e2e_migration/test_migration.py -v -k mysql"
    
    echo "Stopping MySQL..."
    cd e2e_migration/test_mysql
    docker compose down -v
    cd ../..
}

run_postgresql_tests() {
    echo "Starting PostgreSQL..."
    
    # Start PostgreSQL using docker compose
    cd e2e_migration/test_postgresql
    docker compose up -d
    
    if ! wait_for_service "localhost" 5432 "PostgreSQL"; then
        docker compose down -v 2>/dev/null || true
        cd ../..
        return 1
    fi
    
    echo "Running PostgreSQL migration tests..."
    cd ../..
    docker exec -i -u memgraph "$MAGE_CONTAINER" bash -c "cd /mage && python3 -m pytest e2e_migration/test_migration.py -v -k postgresql"
    
    echo "Stopping PostgreSQL..."
    cd e2e_migration/test_postgresql
    docker compose down -v
    cd ../..
}

# Main execution
if [ -z "$TEST_FILTER" ] || [ "$TEST_FILTER" = "mysql" ]; then
    run_mysql_tests
fi

if [ -z "$TEST_FILTER" ] || [ "$TEST_FILTER" = "postgresql" ]; then
    run_postgresql_tests
fi
