#!/bin/bash -e

# Container names and images (set via command line arguments)
MAGE_CONTAINER=""
MYSQL_CONTAINER=""
POSTGRESQL_CONTAINER=""
MYSQL_IMAGE=""
POSTGRESQL_IMAGE=""

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
        --mysql-image)
            MYSQL_IMAGE="$2"
            shift 2
            ;;
        --postgresql-image)
            POSTGRESQL_IMAGE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [-k FILTER] --mage-container CONTAINER --mysql-container CONTAINER --postgresql-container CONTAINER [--mysql-image IMAGE] [--postgresql-image IMAGE]"
            echo "  -k FILTER                    Filter tests by database type (e.g., 'mysql', 'postgresql')"
            echo "  --mage-container NAME        MAGE container name (required)"
            echo "  --mysql-container NAME       MySQL container name (required if mysql tests are run)"
            echo "  --mysql-image IMAGE          MySQL image (required if mysql tests are run)"
            echo "  --postgresql-container NAME  PostgreSQL container name (required if postgresql tests are run)"
            echo "  --postgresql-image IMAGE     PostgreSQL image (required if postgresql tests are run)"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MAGE_CONTAINER" ] || [ -z "$MYSQL_CONTAINER" ] || [ -z "$POSTGRESQL_CONTAINER" ]; then
    echo "Error: All container names are required"
    echo "Usage: $0 --mage-container CONTAINER --mysql-container CONTAINER --postgresql-container CONTAINER"
    exit 1
fi

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
    
    # Start MySQL using docker compose with inline environment variables
    cd e2e_migration/test_mysql
    MYSQL_CONTAINER="$MYSQL_CONTAINER" MYSQL_IMAGE="$MYSQL_IMAGE" docker compose up -d
    
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
    
    # Start PostgreSQL using docker compose with inline environment variables
    cd e2e_migration/test_postgresql
    POSTGRESQL_CONTAINER="$POSTGRESQL_CONTAINER" POSTGRESQL_IMAGE="$POSTGRESQL_IMAGE" docker compose up -d
    
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
