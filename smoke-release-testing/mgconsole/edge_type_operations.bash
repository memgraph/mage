#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_edge_type_operations() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Edge Type Operations"

  echo "WITH {my_edge_type: \"KNOWS\"} as x CREATE ()-[:x.my_edge_type]->() RETURN x; MATCH (n) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}
