#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_ttl() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Time-to-live TTL"

  echo "ENABLE TTL EVERY '1d' AT '00:00:00';" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port # TODO(gitbuda): Skip TTL already running error.
  echo "CREATE GLOBAL EDGE INDEX ON :(ttl);" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}
