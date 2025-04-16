#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_ha_under_k8s() {
  __host="$1"
  __port="$2"
  echo "FEATURE: High-availability Automatic Failover"

  echo "SHOW INSTANCES;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Take a look at session_trace.bash for the v1 implementation of binary-docker picker.
  # TODO(gitbuda): Run HA cluster here (at the moment manually run k8s.bash).
  # TODO(gitbuda): Figure out how to properly port forward from the bash script.
  # kubectl port-forward memgraph-coordinator-1-0 10000:7687 &
  test_ha_under_k8s localhost 10000
fi
