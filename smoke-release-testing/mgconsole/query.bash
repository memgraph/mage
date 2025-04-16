#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_query() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Cypher query engine"

  echo "SUBFEATURE: Peridic commit"
  echo "MATCH (n) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "UNWIND range(1, 10) as x CALL { WITH x CREATE (n:Label {id: x}) RETURN n } IN TRANSACTIONS OF 1 ROWS;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "MATCH (n) RETURN n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "USING PERIODIC COMMIT 1 MATCH (n) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "MATCH (n) RETURN n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_test_query.logs" test_query
fi
