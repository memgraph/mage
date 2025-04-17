#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_label_operations() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Label Operations"

  echo "WITH {my_labels: [\"Label1\", \"Label2\"]} as x CREATE (n:x.my_labels) RETURN n; MATCH (n) DETACH DELETE n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}
