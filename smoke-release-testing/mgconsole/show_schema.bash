#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_show_schema_info() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Show schema info"

  echo "CREATE (:Node {prop: 1});" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  data=$(echo "SHOW SCHEMA INFO;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port --output-format=csv --csv-doublequote=true)
  schema=$(echo "$data" | sed 1d)
  echo $schema

  # TODO(gitbuda): Implement and pass to the python schema validator.
  # TODO(gitbuda): Try to enable schema during the runtime.
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
  set -e # To make sure the script will return non-0 in case of a failure.
  run_memgraph_binary_and_test "--log-level=TRACE --log-file=mg_test_show_schema_info.logs --schema-info-enabled=True" test_show_schema_info
fi
