#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_spatial() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Spatial data types and functionalities"

  echo "MATCH (n) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "CREATE (n {xy: point({x:1, y:2})}) RETURN n.xy.x;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "CREATE (n {xy: point({x:1, y:2})}) RETURN n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port

  echo "CREATE POINT INDEX ON :School(location);" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SHOW INDEX INFO;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_spatial_test.logs" test_spatial
fi
