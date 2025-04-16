#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_dynamic_algos() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Dynamic Algorithms"

  echo "CALL mg.procedures() YIELD name;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port | grep "online"
}
