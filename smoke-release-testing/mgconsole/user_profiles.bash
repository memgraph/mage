#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_user_profiles() {
  __host="$1"
  __port="$2"
  echo "FEATURE: User Profiles and Resource Limit Constraints"

  echo "SUBFEATURE: Setup test user and profile"
  echo "CREATE USER test_user IDENTIFIED BY 'password123';" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "CREATE PROFILE user_profile LIMIT SESSIONS 2, TRANSACTIONS_MEMORY 10KB;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port --username test_user --password password123

  # TODO(gitbuda): Assign the profile to the user.
  # TODO(gitbuda): Show the profile.

  echo "SUBFEATURE: Test memory limit enforcement"
  # Test query that should work within 10KB limit
  echo "CREATE (n:TestNode {id: 1, data: 'small'});" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  # This one should fail. TODO: Check for it.
  echo "UNWIND range(1, 10000) AS i CREATE (n:MemoryTest {id: i, data: 'Large data string that exceeds 10KB memory limit ' + i}) RETURN count(n);" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port

  echo "SUBFEATURE: Cleanup test data (profile, user, data)"
  echo "DROP USER test_user;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "DROP PROFILE test_profile;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "MATCH (n:TestNode) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "MATCH (n:MemoryTest) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port

  echo "User profiles and resource limit constraints testing completed successfully"
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_test_user_profiles.logs" test_user_profiles
fi
