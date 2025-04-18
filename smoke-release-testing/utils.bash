#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# TODO(gitbuda): Write the helper to fail with a nice error message in the case required env variables are not set.

MEMGRAPH_BUILD_PATH="${MEMGRAPH_BUILD_PATH:-/tmp/memgraph/build}"
MEMGRAPH_CONSOLE_BINARY="${MEMGRAPH_CONSOLE_BINARY:-$SCRIPT_DIR/mgconsole.build/build/src/mgconsole}"
# Required env vars to define.
MEMGRAPH_ENTERPRISE_LICENSE="${MEMGRAPH_ENTERPRISE_LICENSE:-provide_licanse_string}"
MEMGRAPH_ORGANIZATION_NAME="${MEMGRAPH_ORGANIZATION_NAME:-provide_organization_name_string}"
MEMGRAPH_LAST_RC_DIRECT_DOCKER_IMAGE_ARM="${MEMGRAPH_LAST_RC_DIRECT_DOCKER_IMAGE_ARM:-provide_https_download_link}"
MEMGRAPH_NEXT_RC_DIRECT_DOCKER_IMAGE_ARM="${MEMGRAPH_NEXT_RC_DIRECT_DOCKER_IMAGE_ARM:-provide_https_download_link}"
MEMGRAPH_LAST_RC_DIRECT_DOCKER_IMAGE_X86="${MEMGRAPH_LAST_RC_DIRECT_DOCKER_IMAGE_X86:-provide_https_download_link}"
MEMGRAPH_NEXT_RC_DIRECT_DOCKER_IMAGE_X86="${MEMGRAPH_NEXT_RC_DIRECT_DOCKER_IMAGE_X86:-provide_https_donwload_link}"
MEMGRAPH_LAST_DOCKERHUB_IMAGE="${MEMGRAPH_LAST_DOCKERHUB_IMAGE:-provide_dockerhub_image_name}"
MEMGRAPH_NEXT_DOCKERHUB_IMAGE="${MEMGRAPH_NEXT_DOCKERHUB_IMAGE:-provide_dockerhub_image_name}"

MEMGRAPH_GENERAL_FLAGS="--telemetry-enabled=false --log-level=TRACE --also-log-to-stderr"
MEMGRAPH_ENTERPRISE_DOCKER_ENVS="-e MEMGRAPH_ENTERPRISE_LICENSE=$MEMGRAPH_ENTERPRISE_LICENSE -e MEMGRAPH_ORGANIZATION_NAME=$MEMGRAPH_ORGANIZATION_NAME"
MEMGRAPH_DOCKER_MOUNT_VOLUME_FLAGS="-v mg_lib:/var/lib/memgraph"
MEMGRAPH_FULL_PROPERTIES_SET="{id:0, name:\"tester\", age:37, height:175.0, merried:true}"
MEMGRAPH_PROPERTY_COMPRESSION_FALGS="--storage-property-store-compression-enabled=true --storage-property-store-compression-level=mid"
MEMGRAPH_HA_COORDINATOR_FALGS="--coordinator-port=10001 --bolt-port=7687 --coordinator-id=1 --experimental-enabled=high-availability --coordinator-hostname=localhost --management-port=11001"
MEMGRAPH_SHOW_SCHEMA_INFO_FLAG="--schema-info-enabled=true"
MEMGRAPH_SESSION_TRACE_FLAG="--query-log-directory=/var/log/memgraph/session_traces"
MEMGRAPH_DEFAULT_HOST="localhost"
MEMGRAPH_DEFAULT_PORT="7687"
MEMGRAPH_LAST_DATA_BOLT_PORT="8000"
MEMGRAPH_LAST_COORDINATOR_BOLT_PORT="8001"
MEMGRAPH_NEXT_DATA_BOLT_PORT="8002"
MEMGRAPH_NEXT_COORDINATOR_BOLT_PORT="8003"
MEMGRAPH_LAST_MONITORING_PORT="9001"
MEMGRAPH_NEXT_MONITORING_PORT="9002"

wait_port() {
  __port="$1"
  while ! nc -z localhost $__port; do
    sleep 0.1
  done
}

wait_for_memgraph() {
  __host=$1
  __port=$2
  while ! echo "return 1;" | $MEMGRAPH_CONSOLE_BINARY --host $__host --port $__port > /dev/null 2>&1; do
    sleep 0.1
  done
}

run_memgraph_binary() {
  # note: printing anything is tricky if this is called under $(...).
  __args="$1"
  cd $MEMGRAPH_BUILD_PATH
  # https://stackoverflow.com/questions/10508843/what-is-dev-null-21
  ./memgraph $__args >> /dev/null 2>&1 &
  echo $!
}

run_memgraph_binary_and_test() {
  __args="$1"
  __test_func_name=$2
  __mg_pid=$(run_memgraph_binary "$__args")
  wait_port $MEMGRAPH_DEFAULT_PORT
  $__test_func_name $MEMGRAPH_DEFAULT_HOST $MEMGRAPH_DEFAULT_PORT
  kill -15 $__mg_pid
}

cleanup_memgraph_binary_processes() {
  set +e # This should be called on script EXIT and not fail.
  pids="$(pgrep -f "\./memgraph")" # Only match ./memgraph.
  if [ ! -z "$pids" ]; then
    kill -15 $pids
  fi
}

pull_dockerhub_last_image() {
  if ! docker image inspect $MEMGRAPH_LAST_DOCKERHUB_IMAGE >/dev/null 2>&1; then
    docker pull $MEMGRAPH_LAST_DOCKERHUB_IMAGE
  fi
}
pull_dockerhub_next_image() {
  if ! docker image inspect $MEMGRAPH_NEXT_DOCKERHUB_IMAGE >/dev/null 2>&1; then
    docker pull $MEMGRAPH_NEXT_DOCKERHUB_IMAGE
  fi
}

pull_RC_last_image() {
  if ! docker image inspect $MEMGRAPH_LAST_DOCKERHUB_IMAGE >/dev/null 2>&1; then
    _url_last="$MEMGRAPH_LAST_RC_DIRECT_DOCKER_IMAGE_X86"
    if [ "$(arch)" == "arm64" ]; then
      _url_last="$MEMGRAPH_LAST_RC_DIRECT_DOCKER_IMAGE_ARM"
    fi
    _filename_last=$(basename $_url_last)
    wget "$_url_last" -O "$SCRIPT_DIR/$_filename_last"
    docker load -i "$SCRIPT_DIR/$_filename_last"
  fi
}

pull_RC_next_image() {
  if ! docker image inspect $MEMGRAPH_NEXT_DOCKERHUB_IMAGE >/dev/null 2>&1; then
    _url_next="$MEMGRAPH_NEXT_RC_DIRECT_DOCKER_IMAGE_X86"
    if [ "$(arch)" == "arm64" ]; then
      _url_next="$MEMGRAPH_NEXT_RC_DIRECT_DOCKER_IMAGE_ARM"
    fi
    _filename_next=$(basename $_url_next)
    wget "$_url_next" -O "$SCRIPT_DIR/$_filename_next"
    docker load -i "$SCRIPT_DIR/$_filename_next"
  fi
}

pull_docker_images() {
  __how_to_pull_last="$1"
  __how_to_pull_next="$2"
  if [ "$__how_to_pull_last" == "RC" ]; then
    pull_RC_last_image
  fi
  if [ "$__how_to_pull_last" == "Dockerhub" ]; then
    pull_dockerhub_last_image
  fi
  if [ "$__how_to_pull_next" == "RC" ]; then
    pull_RC_next_image
  fi
  if [ "$__how_to_pull_next" == "Dockerhub" ]; then
    pull_dockerhub_next_image
  fi
}

run_memgraph_last_dockerhub_container() {
  if [ ! "$(docker ps -q -f name=memgraph_last_data)" ]; then
    docker run -d --rm -p $MEMGRAPH_LAST_DATA_BOLT_PORT:7687 -p $MEMGRAPH_LAST_MONITORING_PORT:9091 \
      --name memgraph_last_data \
      $MEMGRAPH_LAST_DOCKERHUB_IMAGE $MEMGRAPH_GENERAL_FLAGS
  fi
}

run_memgraph_next_dockerhub_container() {
  if [ ! "$(docker ps -q -f name=memgraph_next_data)" ]; then
    docker run -d --rm -p $MEMGRAPH_NEXT_DATA_BOLT_PORT:7687 -p $MEMGRAPH_NEXT_MONITORING_PORT:9091 \
      --name memgraph_next_data \
      $MEMGRAPH_ENTERPRISE_DOCKER_ENVS $MEMGRAPH_NEXT_DOCKERHUB_IMAGE $MEMGRAPH_GENERAL_FLAGS \
      $MEMGRAPH_PROPERTY_COMPRESSION_FALGS $MEMGRAPH_SHOW_SCHEMA_INFO_FLAG
  fi
}

run_memgraph_last_dockerhub_container_with_volume() {
  docker run -d --rm -p $MEMGRAPH_LAST_DATA_BOLT_PORT:7687 $MEMGRAPH_DOCKER_MOUNT_VOLUME_FLAGS \
    --name memgraph_last_data $MEMGRAPH_LAST_DOCKERHUB_IMAGE $MEMGRAPH_GENERAL_FLAGS
}

run_memgraph_next_dockerhub_container_with_volume() {
  docker run -d --rm -p $MEMGRAPH_NEXT_DATA_BOLT_PORT:7687 $MEMGRAPH_DOCKER_MOUNT_VOLUME_FLAGS \
    --name memgraph_next_data $ENTERPRISE_DOCKER_ENVS_UNLIMITED $MEMGRAPH_NEXT_DOCKERHUB_IMAGE $MEMGRAPH_GENERAL_FLAGS $MEMGRAPH_PROPERTY_COMPRESSION_FALGS
}

run_memgraph_coordinator_next_dockerhub_container() {
  docker run -d --rm -p $MEMGRAPH_NEXT_COORDINATOR_BOLT_PORT:7687 --name memgraph_next_coordinator \
    $MEMGRAPH_ENTERPRISE_DOCKER_ENVS $MEMGRAPH_NEXT_DOCKERHUB_IMAGE $MEMGRAPH_GENERAL_FLAGS \
    $MEMGRAPH_PROPERTY_COMPRESSION_FALGS $MEMGRAPH_HA_COORDINATOR_FALGS
}

run_memgraph_docker_containers() {
  __how_to_pull_last="$1"
  __how_to_pull_next="$2"
  pull_docker_images "$__how_to_pull_last" "$__how_to_pull_next"
  run_memgraph_last_dockerhub_container
  run_memgraph_next_dockerhub_container
}

docker_stop_if_there() {
  container_name="$1"
  if [ "$(docker ps -q -f name=$container_name)" ]; then
    docker stop $container_name
    docker rm $container_name || true # If container is started with --rm if will automatically get deleted.
  fi
}

cleanup_docker() {
  docker_stop_if_there memgraph_last_data || true
  docker_stop_if_there memgraph_next_data || true
  docker_stop_if_there memgraph_next_coordinator || true
}

cleanup_docker_exit() {
  ARG=$?
  cleanup_docker
  exit $ARG
}

spinup_and_cleanup_memgraph_dockers() {
  __how_to_pull_last="$1"
  __how_to_pull_next="$2"
  cleanup_docker # Run to stop and previously running containers.
  run_memgraph_docker_containers "$__how_to_pull_last" "$__how_to_pull_next"
  trap cleanup_docker_exit EXIT
}
