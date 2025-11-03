#!/bin/bash
set -euo pipefail

CONTAINER_NAME=mgbuild
RUN_RUST_TESTS=true
RUN_CPP_TESTS=true
RUN_PYTHON_TESTS=true
# RUN_E2E_TESTS=true
# RUN_E2E_CORRECTNESS_TESTS=true
# RUN_E2E_MIGRATION_TESTS=true
# MEMGRAPH_NETWORK=${MEMGRAPH_NETWORK:-memgraph_test_network}
# NEO4J_CONTAINER=${NEO4J_CONTAINER:-neo4j_test}
# MYSQL_CONTAINER=${MYSQL_CONTAINER:-mysql_test}
# POSTGRESQL_CONTAINER=${POSTGRESQL_CONTAINER:-postgresql_test}
while [[ $# -gt 0 ]]; do
  case $1 in
    --container-name)
      CONTAINER_NAME=$2
      shift 2
    ;;  
    --skip-rust-tests)
      RUN_RUST_TESTS=false
      shift 1
    ;;  
    --skip-cpp-tests)
      RUN_CPP_TESTS=false
      shift 1
    ;;  
    --skip-python-tests)
      RUN_PYTHON_TESTS=false
      shift 1
    ;;  
    # --skip-e2e-tests)
    #   RUN_E2E_TESTS=false
    #   shift 1
    # ;;  
    # --skip-e2e-correctness-tests)
    #   RUN_E2E_CORRECTNESS_TESTS=false
    #   shift 1
    # ;;  
    # --skip-e2e-migration-tests)
    #   RUN_E2E_MIGRATION_TESTS=false
    #   shift 1
    # ;;  
    *)
      echo "Unknown option: $1"
      exit 1
    ;;  
  esac
done

cleanup() {
  local exit_code=${1:-$?}
  if [[ $exit_code -ne 0 ]]; then
    echo -e "\033[1;31mFailed to run tests\033[0m"
  fi
  echo -e "\033[1;32mStopping containers and network\033[0m"
  docker stop $CONTAINER_NAME || true
  docker rm $CONTAINER_NAME || true
  # docker stop $NEO4J_CONTAINER || true
  # docker rm $NEO4J_CONTAINER || true
  # docker stop $MYSQL_CONTAINER || true
  # docker rm $MYSQL_CONTAINER || true
  # docker stop $POSTGRESQL_CONTAINER || true
  # docker rm $POSTGRESQL_CONTAINER || true
  # docker network rm $MEMGRAPH_NETWORK || true
  exit $exit_code
}

trap cleanup ERR EXIT

echo -e "\033[1;32mRunning tests in container: $CONTAINER_NAME\033[0m"

# echo -e "\033[1;32mCreating network\033[0m"
# docker network create $MEMGRAPH_NETWORK || true
# docker network connect $MEMGRAPH_NETWORK $CONTAINER_NAME

if [[ "$RUN_RUST_TESTS" == true ]]; then
  echo -e "\033[1;32mRunning Rust tests\033[0m"
  docker exec -i -u root $CONTAINER_NAME bash -c "apt-get update \
  && apt-get install -y libpython${PY_VERSION:-$(python3 --version | sed 's/Python //')} \
  libcurl4 libssl-dev openssl build-essential cmake curl g++ python3  \
  python3-pip python3-setuptools python3-dev clang git unixodbc-dev \
  libboost-all-dev uuid-dev gdb procps libc6-dbg libxmlsec1-dev xmlsec1 \
  --no-install-recommends"
  docker exec -i -u mg $CONTAINER_NAME bash -c "source \$HOME/.cargo/env && cd \$HOME/mage/rust/rsmgp-sys && cargo fmt -- --check && RUST_BACKTRACE=1 cargo test"
fi

if [[ "$RUN_CPP_TESTS" == true ]]; then
  echo -e "\033[1;32mRunning C++ tests\033[0m"
  docker exec -i -u mg $CONTAINER_NAME bash -c "cd \$HOME/mage/cpp/build/ && ctest --output-on-failure -j$CORE_COUNT"
fi

if [[ "$RUN_PYTHON_TESTS" == true ]]; then
  echo -e "\033[1;32mRunning Python tests\033[0m"
  docker exec -i -u mg $CONTAINER_NAME bash -c "cd \$HOME/mage/python/ && python3 -m pytest ."
fi

# if [[ "$RUN_E2E_TESTS" == true ]]; then
#   echo -e "\033[1;32mRunning E2E tests\033[0m"
#   docker exec -i -u memgraph $CONTAINER_NAME bash -c "cd \$HOME/mage/e2e/ && python3 -m pytest . -k 'not cugraph and not embeddings_test-test_cuda_compute'"
# fi

# if [[ "$RUN_E2E_CORRECTNESS_TESTS" == true ]]; then
#   echo -e "\033[1;32mRunning E2E correctness tests\033[0m"
#   docker exec -i -u memgraph $CONTAINER_NAME bash -c "cd \$HOME/mage/e2e/ && python3 -m pytest . -k 'not cugraph and not embeddings_test-test_cuda_compute'"
#   # always stop and remove neo4j container
#   docker stop $NEO4J_CONTAINER || true
#   docker rm $NEO4J_CONTAINER || true
# fi

# if [[ "$RUN_E2E_MIGRATION_TESTS" == true ]]; then
#   echo -e "\033[1;32mRunning E2E migration tests\033[0m"
#   docker exec -i -u memgraph $CONTAINER_NAME bash -c "cd \$HOME/mage/e2e/ && python3 -m pytest . -k 'not cugraph and not embeddings_test-test_cuda_compute'"
# fi
