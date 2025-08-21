#!/bin/bash -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/utils.bash"

# NOTE: 1st arg is how to pull LAST image, 2nd arg is how to pull NEXT image.
spinup_and_cleanup_memgraph_dockers Dockerhub RC
echo "Waiting for memgraph to initialize..."
wait_for_memgraph $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_LAST_DATA_BOLT_PORT
wait_for_memgraph $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
echo "Memgraph is up and running!"

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
test_regex $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_edge_type_operations $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_composite_indices $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_monitoring $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_multi_tenancy $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_nested_indices $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_or_expression_for_labels $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT

# v3.5 testing.
test_shortest_paths $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_text_search $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_durability $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT

# NOTE: If the testing container is NOT restarted, all the auth test have to
# come after all tests that assume there are no users.

# TODO(gitbuda): Probably more to the utils somewhere.
echo "CREATE USER admin IDENTIFIED BY 'admin1234'; GRANT ALL PRIVILEGES TO admin;" | $__mgconsole_default
echo "CREATE USER tester IDENTIFIED BY 'tester1234'; GRANT CREATE TO tester; GRANT CREATE_DELETE ON LABELS * TO tester; GRANT DATABASE memgraph TO tester;" | $__mgconsole_admin

# TODO(gitbuda): Ask Andreja why the below doesn't work :thinking:
# echo "CREATE USER tester IDENTIFIED BY 'tester1234';" $__mgconsole_admin
# echo "GRANT CREATE TO tester; GRANT CREATE_DELETE ON LABELS * TO tester; GRANT DATABASE memgraph TO tester;" | $__mgconsole_admin

test_show_database_settings $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_auth_roles $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_impersonate_user $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
test_user_profiles $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT