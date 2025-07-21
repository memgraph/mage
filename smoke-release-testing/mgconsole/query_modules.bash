#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_query_modules() {
  __host="$1"
  __port="$2"
  echo "FEATURE: All MAGE query modules"

  # TODO(gitbuda): What are the 3 added modules? -> Probably the migration modules.
  echo "CALL mg.procedures() YIELD * RETURN count(*) AS cnt;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port --output-format=csv | python3 $SCRIPT_DIR/validator.py first_as_int -f cnt -e 312

  echo "CREATE (a), (b), (c), (d), (a)-[:ET]->(b), (c)-[:ET]->(d);" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "CALL leiden_community_detection.get() YIELD * RETURN communities, community_id, node;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.
  # TODO/NOTE: Seems like SCRIPT_DIR gets override on the source call! -> FIX
  # TODO(gitbuda): When memgraph is started manually -> logs from basic.cpp module are visible -> under the below logs they are not because it's logs vs output -> FIX
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_test_query_modules.logs --query-modules-directory=$SCRIPT_DIR/query_modules/dist --also-log-to-stderr" test_query_modules
fi
