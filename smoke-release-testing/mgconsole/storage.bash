#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_storage() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Storage: IN_MEMORY_TRANSACTIONAL"

  echo "SUBFEATURE: Property compression"
  echo "CREATE (n:Label $MEMGRAPH_FULL_PROPERTIES_SET)-[:Edge $MEMGRAPH_FULL_PROPERTIES_SET]->(:Label);" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "MATCH (n) RETURN n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_test_storage.logs --storage-properties-on-edges=True" test_storage
fi
