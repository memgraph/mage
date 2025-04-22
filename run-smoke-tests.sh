#!/bin/bash
source /media/raid/Work/github/.env
CONTAINER_NAME="ubuntu_dind"
NEXT_IMAGE="memgraph/memgraph-mage:3.1.1-memgraph-3.1.1"
LAST_IMAGE="memgraph/memgraph-mage:3.0-memgraph-3.0"

docker build -t ubuntu:24.04-dind --file Dockerfile.smoke .

docker run -d --rm  --privileged \
    --name $CONTAINER_NAME \
    -e container=docker \
    -p 2375:2375 \
    --tmpfs /run \
    --tmpfs /run/lock \
    --tmpfs /tmp \
    -v u24-data:/var/lib/docker \
    -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
    --cgroupns=host \
    ubuntu:24.04-dind

docker exec -i -u root $CONTAINER_NAME bash -c "docker pull ${NEXT_IMAGE}"
docker exec -i -u root $CONTAINER_NAME bash -c "docker pull ${LAST_IMAGE}"
docker exec -i -u root $CONTAINER_NAME bash -c "cd /mage/smoke-release-testing && ./init_workflow.bash"
docker exec -i -u root $CONTAINER_NAME bash -c "pip install -r /mage/smoke-release-testing/requirements.txt --break-system-packages"
docker exec -i -u root \
    -e MEMGRAPH_ENTERPRISE_LICENSE="$MEMGRAPH_ENTERPRISE_LICENSE" \
    -e MEMGRAPH_ORGANIZATION_NAME="$MEMGRAPH_ORGANIZATION_NAME" \
    -e MEMGRAPH_LAST_DOCKERHUB_IMAGE="$LAST_IMAGE" \
    -e MEMGRAPH_NEXT_DOCKERHUB_IMAGE="$NEXT_IMAGE" \
    $CONTAINER_NAME \
    bash -c "cd /mage/smoke-release-testing && ./test.bash"

docker stop $CONTAINER_NAME