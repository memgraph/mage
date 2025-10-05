#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

sudo sysctl -w vm.max_map_count=1048576

# Copy the desired module, convenient to change the mage code and just run it
# under the container.
cp $SCRIPT_DIR/../../python/embed_worker/embed_worker.py ./
cp $SCRIPT_DIR/../../python/embeddings.py ./

docker run --name=embeddings --rm --gpus=all --network host -p 7687:7687 \
  -v $SCRIPT_DIR:/app memgraph/memgraph-mage:gpu \
  --schema-info-enabled=True --telemetry-enabled=False \
  --also-log-to-stderr --log-level=TRACE

# TODO(gitbuda): Run the container setup.
