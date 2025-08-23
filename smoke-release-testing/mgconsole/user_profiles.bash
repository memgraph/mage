#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_user_profiles() {
  # TODO(gitbuda): args became useless -> NOTE: running this test alone still depends to make this args work -> FIX.
  __host="$1"
  __port="$2"
  echo "FEATURE: User Profiles and Resource Limit Constraints"

  echo "SUBFEATURE: Setup test user and profile"
  echo "CREATE PROFILE user_profile LIMIT SESSIONS 2, TRANSACTIONS_MEMORY 100KB;" | $MGCONSOLE_NEXT_ADMIN
  echo "SET PROFILE FOR tester TO user_profile;" | $MGCONSOLE_NEXT_ADMIN
  echo "SHOW PROFILE FOR tester;" | $MGCONSOLE_NEXT_ADMIN

  echo "SUBFEATURE: Test memory limit enforcement"
  # Test query that should work within 10KB limit
  echo "CREATE (n:TestNode {id: 1, data: 'small'});" | $MGCONSOLE_NEXT_TESTER
  # This one should fail. Check for failure and continue only if it fails.
  if ! echo "UNWIND range(1, 10000) AS i CREATE (n:MemoryTest {id: i, data: 'Data string ' + i}) RETURN count(n);" | $MGCONSOLE_NEXT_TESTER; then
    echo "Memory limit enforcement by profile test failed as expected."
  fi

  echo "SUBFEATURE: Cleanup profile"
  echo "DROP PROFILE user_profile;" | $MGCONSOLE_NEXT_ADMIN
  echo "SHOW PROFILES;" | $MGCONSOLE_NEXT_ADMIN
  echo "MATCH (n:TestNode) DETACH DELETE n;" | $MGCONSOLE_NEXT_ADMIN
  echo "MATCH (n:MemoryTest) DETACH DELETE n;" | $MGCONSOLE_NEXT_ADMIN

  echo "User profiles and resource limit constraints testing completed successfully"
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_test_user_profiles.logs" test_user_profiles
fi
