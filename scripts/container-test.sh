#!/bin/bash
set -euo pipefail

CONTAINER_NAME=mgbuild
CI=false
CACHE_PRESENT=false
CUDA=false
ARCH=amd64
while [[ $# -gt 0 ]]; do
  case $1 in
    --container-name)
      CONTAINER_NAME=$2
      shift 2
    ;;  
    --ci)
      CI=true
      shift
      ;;
    --cache-present)
      CACHE_PRESENT=$2
      shift 2
      ;;
    --cuda)
      CUDA=$2
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
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
  exit $exit_code
}

trap cleanup ERR EXIT

echo -e "\033[1;32mRunning tests in container: $CONTAINER_NAME\033[0m"

echo -e "\033[1;32mRunning Rust tests\033[0m"
docker exec -i -u root $CONTAINER_NAME bash -c "apt-get update \
&& apt-get install -y clang --no-install-recommends"
docker exec -i -u mg $CONTAINER_NAME bash -c "source \$HOME/.cargo/env && cd \$HOME/mage/rust/rsmgp-sys && cargo fmt -- --check && RUST_BACKTRACE=1 cargo test"


echo -e "\033[1;32mRunning C++ tests\033[0m"
docker exec -i -u mg $CONTAINER_NAME bash -c "cd \$HOME/mage/cpp/build/ && ctest --output-on-failure -j\$(nproc)"


echo -e "\033[1;32mRunning Python tests\033[0m"
if [[ "$CUDA" == true ]]; then
  requirements_file="requirements-gpu.txt"
else
  requirements_file="requirements.txt"
fi
docker cp python/$requirements_file $CONTAINER_NAME:/tmp/$requirements_file
docker cp cpp/memgraph/src/auth/reference_modules/requirements.txt $CONTAINER_NAME:/tmp/auth_module-requirements.txt
docker exec -i -u mg $CONTAINER_NAME bash -c "cd \$HOME/mage/ && \
  ./scripts/install_python_requirements.sh --ci --cache-present $CACHE_PRESENT --cuda $CUDA --arch $ARCH && \
  pip install -r \$HOME/mage/python/tests/requirements.txt --break-system-packages"
docker exec -i -u mg $CONTAINER_NAME bash -c "cd \$HOME/mage/python/ && python3 -m pytest ."

