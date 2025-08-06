#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_vector_search() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Vector Search"

  echo "CREATE VECTOR INDEX vsi ON :Label(embedding) WITH CONFIG {\"dimension\":2, \"capacity\": 10};" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "CREATE VECTOR EDGE INDEX etvsi ON :EdgeType(embedding) WITH CONFIG {\"dimension\": 256, \"capacity\": 1000};" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port

  echo "CALL vector_search.show_index_info() YIELD * RETURN *;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "SHOW VECTOR INDEX INFO;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}
