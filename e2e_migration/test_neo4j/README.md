# Neo4j Migration Tests

This directory contains tests for migrating data from Neo4j to Memgraph using the `migrate.neo4j()` procedure.

## Structure

- `test/successful/` - Tests that should succeed
- `test/unsuccessful/` - Tests that should fail (unsupported data types)
- `docker-compose.yml` - Docker configuration for Neo4j container

## Supported Data Types

The following Neo4j data types are tested for successful migration:

### Basic Types
- **Boolean**: `true`, `false`
- **Integer**: Various integer values including edge cases
- **Float**: Floating-point numbers
- **String**: Text data including Unicode and special characters
- **Null**: Null values

### Temporal Types
- **Date**: Date values (YYYY-MM-DD format)
- **DateTime**: Date and time values (ISO 8601 format)
- **LocalDateTime**: Local date and time values

### Complex Types
- **List**: Arrays of various data types
- **Map**: Key-value pairs and nested structures

## Unsupported Data Types

The following Neo4j data types are expected to fail migration:

- **Point**: Spatial point data
- **Time**: Time-only values (without date)
- **LocalTime**: Local time values
- **Duration**: Duration values

## Test Approach

The tests use direct Cypher expressions to test data type migration:
- `RETURN true` for boolean values
- `RETURN 42` for integer values
- `RETURN 3.14159` for float values
- `RETURN "Hello World"` for string values
- `RETURN date("2023-12-25")` for date values
- `RETURN datetime("2023-12-25T14:30:00")` for datetime values
- `RETURN [1, 2, 3]` for list values
- `RETURN {key: "value"}` for map values
- `RETURN null` for null values

This approach is simpler and more direct than creating and querying test nodes.

## Running Tests

The tests are automatically discovered by the migration test runner. To run only Neo4j tests:

```bash
python3 run_migration_tests.py -k neo4j
```

## Docker Configuration

The Neo4j container is configured with:
- No authentication required
- APOC plugin enabled
- File import/export enabled
- Health checks for container readiness
