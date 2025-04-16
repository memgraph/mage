#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_impersonate_user() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Impersonate User"

  echo "CREATE USER admin; GRANT IMPERSONATE_USER * TO admin;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}
