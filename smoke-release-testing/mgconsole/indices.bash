#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_composite_indices() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Label Property Composit Index"

  echo "CREATE INDEX ON :Label(prop1, prop2);" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "CREATE (:Label {prop1:0, prop2: 1});" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "EXPLAIN MATCH (n:Label {prop1:0, prop2: 1}) RETURN n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port | grep -E "ScanAllByLabelProperties \(n :Label \{prop1, prop2\}\)"
}

test_nested_indices() {
}
