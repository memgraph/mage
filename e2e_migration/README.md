# E2E Migration Testing Framework

This directory contains end-to-end tests for the migration functionality in MAGE. The tests verify that data can be successfully migrated from various database systems (MySQL, PostgreSQL, etc.) to Memgraph using the `migrate` module.

## Overview

The e2e_migration framework provides:
- **Database Setup**: Automated setup of source databases (MySQL, PostgreSQL) with test data
- **Migration Testing**: Comprehensive tests for data migration using the `migrate` module
- **Validation**: Verification that migrated data matches the source data
- **Error Handling**: Tests for various error scenarios
- **Database Agnostic**: Framework supports multiple database types through test_* directories

## Structure

```
e2e_migration/
├── pytest.ini                 # Pytest settings
├── migration_utils.py          # Utility functions for testing
├── test_migration.py           # Main test file (database-agnostic)
├── run_migration_tests.py      # Test execution script (auto-discovers test_* dirs)
├── test_mysql/                # MySQL-specific test directory
│   ├── docker-compose.yml     # MySQL service orchestration (official mysql:8.0 image)
│   ├── data/                  # Database initialization scripts
│   │   ├── init.sql           # Schema creation
│   │   └── sample_data.sql    # Test data insertion
│   └── test/                  # MySQL test cases
│       └── test_migration.yml # MySQL migration query and expected output
└── test_postgresql/           # PostgreSQL-specific test directory
    ├── docker-compose.yml     # PostgreSQL service orchestration (official postgres:15 image)
    ├── data/                  # Database initialization scripts
    │   ├── init.sql           # Schema creation
    │   └── sample_data.sql    # Test data insertion
    └── test/                  # PostgreSQL test cases
        └── test_migration.yml # PostgreSQL migration query and expected output
```

## Test Structure

The migration tests follow the same pattern as the existing e2e tests:

### Test Configuration Format
Each `test_migration.yml` file contains:
- **query**: The migration procedure call that returns data (e.g., `CALL migrate.mysql(...) YIELD row RETURN row.*`)
- **output**: Expected results for validation (list of dictionaries with field names and expected values)

### Example Test Configuration
```yaml
query: |
  CALL migrate.mysql('dummy_table', {...}) YIELD row
  RETURN row.id, row.name, row.value
  ORDER BY row.id

output:
  - id: 1
    name: "test"
    value: 42
```

### Validation Process
1. Execute the migration query
2. Compare actual results with expected output
3. Handle data type conversions (float tolerance, string conversion, etc.)
4. Assert all fields match expected values

### CI/CD Integration

The migration tests are integrated into the GitHub Actions workflow (`reusable_test.yml`):
- **Memgraph**: Runs in the main MAGE container (no separate Memgraph container needed)
- **Source Databases**: Each test starts its own database container using official images
- **Network**: Tests connect to source databases via container names
- **Execution**: Tests run inside the MAGE container with access to both Memgraph and source databases

### Simplified Setup

The framework uses fixed, predictable values for maximum simplicity:
- **Memgraph Connection**: Always `localhost:7687`
- **Test Configuration Path**: Always `test_{db_type}/test/test_migration.yml`
- **No Configuration Files**: All values are hardcoded for consistency
- **Official Database Images**: `mysql:8.0` and `postgres:15`
- **No Custom Dockerfiles**: Initialization scripts are mounted via volumes
- **Fast Startup**: Official images are optimized and cached

Each database test includes:
- **dummy_table**: A single table with one row containing all common data types for that database:

### MySQL Test Data
- **Numeric types**: TINYINT, SMALLINT, MEDIUMINT, BIGINT, DECIMAL, FLOAT, DOUBLE, BIT
- **Date/Time types**: DATE, TIME, DATETIME, TIMESTAMP, YEAR
- **String types**: CHAR, VARCHAR, BINARY, VARBINARY, TINYBLOB, TINYTEXT, BLOB, TEXT, MEDIUMBLOB, MEDIUMTEXT, LONGBLOB, LONGTEXT, ENUM, SET
- **JSON type**: JSON
- **Boolean types**: BOOLEAN, BOOL

### PostgreSQL Test Data
- **Numeric types**: SMALLINT, INTEGER, BIGINT, DECIMAL, NUMERIC, REAL, DOUBLE PRECISION, MONEY
- **Date/Time types**: TIMESTAMP, TIMESTAMP WITH TIME ZONE, DATE, TIME, TIME WITH TIME ZONE, INTERVAL
- **String types**: CHAR, VARCHAR, TEXT, BYTEA
- **JSON types**: JSON, JSONB
- **Boolean types**: BOOLEAN
- **Geometric types**: POINT, LINE, LSEG, BOX, PATH, POLYGON, CIRCLE
- **Network types**: CIDR, INET, MACADDR
- **Bit types**: BIT, BIT VARYING
- **Other types**: UUID, XML, ARRAY, RANGE

## Running Tests

### Prerequisites

1. Docker and docker-compose installed
2. Python with pytest installed
3. Required Python packages:
   - `mysql-connector-python` (for MySQL tests)
   - `psycopg2-binary` (for PostgreSQL tests)
   - `gqlalchemy`
   - `pyyaml`

### Quick Start

#### Option 1: Using the main test script (recommended)
```bash
# From the mage root directory
./test_e2e_migration.py
```

#### Option 2: Using the internal runner
```bash
# From the e2e_migration directory
cd e2e_migration
./run_tests.sh
# or
python3 run_migration_tests.py
```

#### Option 3: Using pytest directly
```bash
# From the mage root directory
python3 -m pytest e2e_migration/ -v
```

The scripts automatically:
- Discover all `test_*` directories
- Start services for each database system
- Run migration tests
- Clean up services

### Manual Setup

1. **Start Services** (for MySQL):
   ```bash
   cd test_mysql
   docker-compose up -d
   ```

2. **Wait for Services** (30 seconds):
   ```bash
   docker-compose ps
   ```

3. **Run Tests**:
   ```bash
   cd ..
   pytest test_migration.py -v --tb=short \
       --mysql-host=localhost \
       --mysql-port=3306 \
       --mysql-user=memgraph \
       --mysql-password=memgraph_password \
       --mysql-database=test_db \
       --memgraph-port=7687
   ```

4. **Cleanup**:
   ```bash
   cd test_mysql
   docker-compose down -v
   ```

## Test Cases

### 1. Dummy Table Migration (`test_dummy_table_migration`)
- Migrates a single row with all MySQL data types from MySQL to Memgraph
- Creates a single DummyNode with all properties
- Validates all MySQL data type conversions:
  - **Numeric types**: TINYINT, SMALLINT, MEDIUMINT, BIGINT, DECIMAL→float, FLOAT, DOUBLE, BIT
  - **Date/Time types**: DATE, TIME, DATETIME, TIMESTAMP, YEAR (converted to strings)
  - **String types**: CHAR, VARCHAR, BINARY, VARBINARY, TINYBLOB, TINYTEXT, BLOB, TEXT, MEDIUMBLOB, MEDIUMTEXT, LONGBLOB, LONGTEXT, ENUM, SET
  - **JSON type**: JSON (preserved as-is)
  - **Boolean types**: BOOLEAN, BOOL
- Tests comprehensive data type preservation and conversion

## Configuration

### MySQL Configuration
- Host: localhost
- Port: 3306
- User: memgraph
- Password: memgraph_password
- Database: test_db

### Memgraph Configuration
- Host: localhost
- Port: 7687

### Command Line Options

The `test_e2e_migration.py` script supports various options:

```bash
# Run all migration tests
./test_e2e_migration.py

# Run only MySQL tests
./test_e2e_migration.py -k mysql

# Run only PostgreSQL tests
./test_e2e_migration.py -k postgresql

# Show help
./test_e2e_migration.py --help
```

### Custom Configuration

You can override default settings using pytest options:

```bash
pytest test_migration.py \
    --mysql-host=your-mysql-host \
    --mysql-port=your-mysql-port \
    --mysql-user=your-mysql-user \
    --mysql-password=your-mysql-password \
    --mysql-database=your-mysql-database \
    --memgraph-port=your-memgraph-port
```

## Adding New Database Systems

To add a new database system (e.g., PostgreSQL):

1. **Create Test Directory**:
   ```bash
   mkdir test_postgresql
   ```

2. **Add Database Files**:
   ```bash
   test_postgresql/
   ├── docker-compose.yml     # PostgreSQL service orchestration
   ├── Dockerfile.postgresql  # PostgreSQL container setup
   ├── data/                  # Database initialization scripts
   │   ├── init.sql
   │   └── sample_data.sql
   └── test/                  # Test cases
       ├── test_migration.cyp
       └── test_migration.yml
   ```

3. **Update Test Class**: Add new test methods to `test_migration.py`
4. **Update Documentation**: Update this README with new test descriptions

The `run_tests.sh` script will automatically discover and run tests for any `test_*` directory.

## Adding New Tests

1. **Create Test Files**: Add `.cyp` and `.yml` files in the appropriate test directory
2. **Update Test Class**: Add new test methods to `test_migration.py`
3. **Update Documentation**: Update this README with new test descriptions

### Test File Format

**Cypher File** (`.cyp`):
```cypher
// Test description
CALL migrate.mysql('table_name', {
    host: 'localhost',
    port: 3306,
    user: 'memgraph',
    password: 'memgraph_password',
    database: 'test_db'
})
YIELD row
CREATE (n:Node {property: row.property});
```

**YAML File** (`.yml`):
```yaml
query: |
  CALL migrate.mysql('table_name', {...})
  YIELD row
  CREATE (n:Node {property: row.property});

mysql_query: "SELECT * FROM table_name ORDER BY id"

expected_result:
  - "Description of expected results"
  - "Validation criteria"

validation_queries:
  - "MATCH (n:Node) RETURN count(n) as count"
  - "MATCH (n:Node) RETURN n.property ORDER BY n.id"
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure services are running and healthy
2. **Authentication Failed**: Check MySQL credentials
3. **Test Timeouts**: Increase wait time for service startup
4. **Data Mismatch**: Verify test data is properly loaded

### Debug Mode

Run tests with debug output:
```bash
pytest test_migration.py -v -s --log-cli-level=DEBUG
```

### Service Logs

Check service logs:
```bash
docker-compose logs mysql
docker-compose logs memgraph
```

## Contributing

When adding new migration tests:

1. Follow the existing test structure
2. Add comprehensive validation
3. Test both success and error scenarios
4. Update documentation
5. Ensure tests are deterministic and repeatable
