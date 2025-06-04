#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

# TODO(gitbuda): Introduce default __host and __port values.
test_nested_indices() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Nested Indices"

  echo "
    CREATE (:Product {specifications: {dimensions: {width: 10, height: 20}}});
    CREATE INDEX ON :Product(specifications.dimensions.width);
    RETURN 1;
  " | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  trap cleanup_memgraph_binary_processes EXIT
  set -e
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_nested_indices_test.logs" test_nested_indices
fi