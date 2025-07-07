#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

# TODO(gitbuda): Introduce default __host and __port values.
test_regex() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Regex"

  echo "CREATE (:Hero {name: 'xSPIDERy'});" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "CREATE (:Hero {name: 'test'});" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "MATCH (h:Hero) WHERE h.name =~ \".*SPIDER.+\" RETURN h.name as PotentialSpiderDude ORDER BY PotentialSpiderDude;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_test_regex.logs" test_template
fi
