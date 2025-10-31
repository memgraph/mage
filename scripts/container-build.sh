#!/bin/bash
set -euo pipefail

ARCH=amd64
BUILD_TYPE=Release
CONTAINER_NAME=mgbuild
STOP_CONTAINER=true
while [[ $# -gt 0 ]]; do
  case $1 in
    --arch)
      ARCH=$2
      shift 2
    --build-type)
      BUILD_TYPE=$2
      shift 2
    --container-name)
      CONTAINER_NAME=$2
      shift 2
    --stop-container)
      STOP_CONTAINER=$2
      shift 2
    *)
      echo "Unknown option: $1"
      exit 1
  esac
done

cleanup() {
    local exit_code=${1:-$?}
    # Always clean up on error, or if STOP_CONTAINER is true on normal exit
    if [[ $exit_code -ne 0 ]] || [[ "$STOP_CONTAINER" == true ]]; then
        docker stop $CONTAINER_NAME || true
        docker rm $CONTAINER_NAME || true
    fi
    if [[ $exit_code -ne 0 ]]; then
        # in red bold text
        echo -e "\033[1;31mFailed to build MAGE\033[0m"
    fi
    exit $exit_code
}

trap 'cleanup $?' ERR EXIT


if [[ "$ARCH" == "arm64" ]]; then
    MGBUILD_IMAGE="memgraph/mgbuild:v7_ubuntu-24.04-arm"
else
    MGBUILD_IMAGE=memgraph/mgbuild:v7_ubuntu-24.04
fi

# in bold green
echo -e "\033[1;32mBuilding MAGE - build type: $BUILD_TYPE, arch: $ARCH\033[0m"
echo -e "\033[1;32mUsing base image: $MGBUILD_IMAGE\033[0m"

echo -e "\033[1;32mStarting container\033[0m"
docker run -d --rm --name $CONTAINER_NAME $MGBUILD_IMAGE

echo -e "\033[1;32mCopying repo into container\033[0m"
docker exec -i -u mg $CONTAINER_NAME mkdir -p /home/mg/mage
docker cp . $CONTAINER_NAME:/home/mg/mage
docker exec -i -u root $CONTAINER_NAME bash -c "chown -R mg:mg /home/mg/mage"

echo -e "\033[1;32mBuilding inside container\033[0m"
docker exec -i $CONTAINER_NAME bash -c "cd /home/mg/mage && ./scripts/build.sh $BUILD_TYPE"

echo -e "\033[1;32mCompressing query modules\033[0m"
docker exec -i $CONTAINER_NAME bash -c "cd /home/mg/mage && ./scripts/compress.sh"

echo -e "\033[1;32mCopying compressed query modules\033[0m"
docker cp $CONTAINER_NAME:/home/mg/mage.tar.gz .
