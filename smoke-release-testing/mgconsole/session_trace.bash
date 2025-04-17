#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

test_session_trace() {
  __host="$1"
  __port="$2"
  echo "FEATURE: Inspection"

  echo "SUBFEATURE: Session trace"
  echo "SET SESSION TRACE ON;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "CREATE (n);" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
  echo "MATCH (n) RETURN n;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port
}

# NOTE: No extra functions to not polute the namespace, flags could be different for each mgconsole test.
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  case $1 in
    binary)
      trap cleanup_memgraph_binary_processes EXIT # To make sure cleanup is done.
      set -e # To make sure the script will return non-0 in case of a failure.
      run_memgraph_binary_and_test "--query-log-directory=dir1 --log-level=TRACE --log-file=mg.logs" test_session_trace
    ;;
    docker)
      trap cleanup_docker_exit EXIT
      cleanup_docker
      docker run -d --rm -p $MEMGRAPH_NEXT_DATA_BOLT_PORT:7687 --name memgraph_next_data \
        $ENTERPRISE_DOCKER_ENVS_UNLIMITED $MEMGRAPH_NEXT_DOCKERHUB_IMAGE $MEMGRAPH_GENERAL_FLAGS \
        $MEMGRAPH_PROPERTY_COMPRESSION_FALGS $MEMGRAPH_SHOW_SCHEMA_INFO_FLAG $MEMGRAPH_SESSION_TRACE_FLAG
      sleep 2
      test_session_trace localhost $MEMGRAPH_NEXT_DATA_BOLT_PORT
    ;;
    *)
      exit 1
    ;;
  esac
fi
