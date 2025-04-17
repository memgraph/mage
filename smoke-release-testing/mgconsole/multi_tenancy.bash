#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_multi_tenancy() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Multi-tenancy basic check"

  echo "SHOW DATABASES;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}
