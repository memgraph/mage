#!/bin/bash -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/utils.bash"

# NOTE: Use the below line if you just want to spin up the containers and leave them running.
# run_memgraph_docker_containers RC RC
# NOTE: Use the below line if you want to cleanup the containers after run of this script.
# TODO(gitbuda): The below picking of what to download is super error prone -> improve.
# spinup_and_cleanup_memgraph_dockers Dockerhub Dockerhub
# echo "Waiting for memgraph to initialize..."
# wait_for_memgraph $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
# echo "Memgraph is up and running!"

# source ./mgconsole/vector_search.bash
# test_vector_search $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT

source ./k8s/run.bash
# test_k8s_single
# test_k8s_ha LAST
test_k8s_ha NEXT

# NOTE: Test what's the exit status of the script by using `echo $?`:
#   * if it's == 0 -> all good
#   * if it's != 0 -> something went wrong.
