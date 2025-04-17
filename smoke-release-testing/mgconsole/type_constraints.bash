#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_type_constraints() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Constraints: Data type"
  echo "MATCH (n) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "CREATE CONSTRAINT ON (n:Node) ASSERT n.prop IS TYPED STRING;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port

  set +e
  echo "CREATE (n:Node {prop:23});" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  if [ $? -eq 0 ]; then
    echo "ERROR: Constraint violation not detected."
    exit 1
  fi
  set -e
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.

  # NOTE: The below run required the license key, it should already be set, utils script provides dummy default.
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_test_type_constraints.logs" test_type_constraints
fi
