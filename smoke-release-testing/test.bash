#!/bin/bash -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/utils.bash"

# TODO(gitbuda): Measure the total execution time, it should be under ~10s.
# TODO(gitbuda): Test docker compose.

# NOTE: 1st arg is how to pull LAST image, 2nd arg is how to pull NEXT image.
spinup_and_cleanup_memgraph_dockers DockerHub RC
echo "Waiting for memgraph to initialize..."
wait_for_memgraph $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_LAST_DATA_BOLT_PORT
wait_for_memgraph $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
echo "Memgraph is up and running!"

# Test drivers.
python3 python/gqla.py "3.0.0" $MEMGRAPH_LAST_DATA_BOLT_PORT
python3 python/neo.py "3.0.0" $MEMGRAPH_LAST_DATA_BOLT_PORT
python3 python/gqla.py "3.1.0" $MEMGRAPH_NEXT_DATA_BOLT_PORT
python3 python/neo.py "3.1.0" $MEMGRAPH_NEXT_DATA_BOLT_PORT

# Test features using mgconsole.
for test_file_path in "$SCRIPT_DIR/mgconsole/"*; do
  if [ "$(basename $test_file_path)" == "README.md" ]; then
    continue
  fi
  source $test_file_path
  echo "Loaded $test_file_path..."
done

test_auth_roles $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_basic_auth $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_query $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_query_modules $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
set +e # NOTE: At the time of writing this failed becuase of a bug but the test/config is legit.
       # Remove set +e after fix.
test_session_trace $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
set -e
test_show_schema_info $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_spatial $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_storage $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_streams $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_ttl $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_type_constraints $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_vector_search $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_dynamic_algos $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_functions $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_label_operations $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_edge_type_operations $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_composite_indices $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_monitoring $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_multi_tenancy $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_nested_indices $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
# NOTE: If the testing container is NOT restarted, all the auth test have to
# come after all tests that assume there are no users.
test_impersonate_user $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
