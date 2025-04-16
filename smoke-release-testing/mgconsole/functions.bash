#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_functions() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Built-in functions"

  # Added in v3.1.
  echo "RETURN toSet([1, 2, 1]) AS a;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "RETURN length([1,2,3]) AS a;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "CREATE (a), (b), (a)-[c:Type]->(b) RETURN project([a,b], [c]) AS x; MATCH (n) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}
