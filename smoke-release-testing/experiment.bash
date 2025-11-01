#!/bin/bash -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/utils.bash"

# TODO(gitbuda): Add v3.7 tests.

# NOTE: Use the below line if you just want to spin up the containers and leave them running.
# run_memgraph_docker_containers RC RC
# NOTE: Use the below line if you want to cleanup the containers after run of this script.
spinup_and_cleanup_memgraph_dockers none none
echo "Waiting for memgraph to initialize..."
wait_for_memgraph $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT
echo "Memgraph is up and running!"

source ./mgconsole/user_profiles.bash
test_user_profiles $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_NEXT_DATA_BOLT_PORT

# NOTE: Test what's the exit status of the script by using `echo $?`:
#   * if it's == 0 -> all good
#   * if it's != 0 -> something went wrong.
