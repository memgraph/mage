#!/bin/bash


# Load environment variables from .env file
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi


# Parse command-line arguments
arch="amd64"
build_target="dev"
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --arch)
      arch="$2"
      shift 2
      ;;
    --build-target)
      build_target="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done


echo "Using Memgraph License: $MEMGRAPH_ENTERPRISE_LICENSE"
echo "Using Organization Name: $MEMGRAPH_ORGANIZATION_NAME"

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Prevents errors in a pipeline from being masked

# Define environment variables
PY_VERSION="3.12"
CORE_COUNT="8"
MAGE_CONTAINER="mage"
MEMGRAPH_PORT=7687
NEO4J_PORT=7687
NEO4J_CONTAINER="neo4j_test"
MEMGRAPH_NETWORK="memgraph_test_network"
OS="ubuntu-24.04"
s3_region="eu-west-1"
build_scope="with ML"
memgraph_version="3.0.0"
memgraph_ref="master"
memgraph_ref_update="true"
memgraph_download_link=""

# Checkout repository and submodules
git submodule update --init --recursive
cd cpp/memgraph
git checkout "$memgraph_ref"
if [ "$memgraph_ref_update" == "true" ]; then
    git pull
fi
cd -

# Set up Docker Buildx
docker buildx create --use || true

echo "Setting up and checking memgraph download link..."

if [[ "$arch" == "arm64" ]]; then
    arch_suffix="-aarch64"
else
    arch_suffix=""
fi

if [[ "$build_scope" == "without ML" ]]; then
    DOCKERFILE=Dockerfile.no_ML
else
    DOCKERFILE=Dockerfile.v6mgbuild
fi

if [[ "$arch" == "arm64" ]]; then
    MGBUILD_IMAGE="memgraph/mgbuild:v6_ubuntu-24.04-arm"
else
    MGBUILD_IMAGE="memgraph/mgbuild:v6_ubuntu-24.04"
fi

# Check if the Docker image already exists
if ! docker images | awk '{print $1 ":" $2}' | grep -q "memgraph-mage:$build_target"; then
    echo "Image memgraph-mage:$build_target not found. Building..."
    docker buildx build \
        --tag memgraph-mage:$build_target \
        --target $build_target \
        --platform linux/$arch \
        --file $DOCKERFILE \
        --build-arg MGBUILD_IMAGE="$MGBUILD_IMAGE" \
        --load .
else
    echo "Image memgraph-mage:$build_target already exists. Skipping build."
fi

docker network create "$MEMGRAPH_NETWORK" || true

docker run -d --rm --network "$MEMGRAPH_NETWORK" --name "$MAGE_CONTAINER" \
    -e MEMGRAPH_ENTERPRISE_LICENSE="$MEMGRAPH_ENTERPRISE_LICENSE" \
    -e MEMGRAPH_ORGANIZATION_NAME="$MEMGRAPH_ORGANIZATION_NAME" \
    memgraph-mage:$build_target \
     --telemetry-enabled=False

# Run Rust library tests (only for 'dev' builds)
echo "Rust Library Tests"
if [[ "$build_target" == "dev" ]]; then
    docker exec -i -u memgraph "$MAGE_CONTAINER" bash -c "source ~/.cargo/env && cd /mage/rust/rsmgp-sys && cargo test"
fi

# Run C++ module unit tests
echo "C++ module unit tests"
if [[ "$build_target" == "dev" ]]; then
    docker exec -i -u memgraph "$MAGE_CONTAINER" bash -c "cd /mage/cpp/build/ && ctest --output-on-failure -j$CORE_COUNT"
fi

# Install Python dependencies

docker exec -i -u memgraph "$MAGE_CONTAINER" bash -c "pip install -r /mage/python/tests/requirements.txt --break-system-packages"

# Run Python module tests
if [[ "$build_target" == "dev" && "$build_scope" != "without ML" ]]; then
    echo "Python module tests"
    docker exec -i -u memgraph "$MAGE_CONTAINER" bash -c "cd /mage/python/ && python3 -m pytest ."
fi

# Run End-to-end tests
echo "End-to-end tests"
docker exec -i -u memgraph "$MAGE_CONTAINER" bash -c "cd /mage/e2e/ && python3 -m pytest . -k 'not cugraph'" || true
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "Warning: pytest failed inside the container, but continuing..."
fi

# Run End-to-end correctness tests
./run_e2e_correctness_tests.sh "$MEMGRAPH_PORT" "$NEO4J_PORT" "$NEO4J_CONTAINER" "$MAGE_CONTAINER" "$MEMGRAPH_NETWORK"


docker exec -i "$MAGE_CONTAINER" bash -c 'echo "Using Memgraph License: $MEMGRAPH_ENTERPRISE_LICENSE"; echo "Using Organization Name: $MEMGRAPH_ORGANIZATION_NAME"'


# Cleanup
docker stop "$MAGE_CONTAINER" || true
docker rmi memgraph-mage:$build_target || true
docker network rm "$MEMGRAPH_NETWORK" || true
