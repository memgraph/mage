#!/bin/bash

# This test requires the toolchain archive to be in the directory above mage,
# it also needs to have a plugins directory with the appropriate plugins for 
# neo4j inside it
# run `./toolchain-v7-tests.sh > ../tests.log 2>&1` to save the output to a log


# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")" 
# Change directory to the parent directory of the script's directory
cd $BASE_DIR || { echo "Failed to change directory"; exit 1; }

#check if toolchain exists in /opt
TOOLCHAIN_PATH="/opt/toolchain-v7"
TOOLCHAIN_ACTIVATE="$TOOLCHAIN_PATH/activate"
if [ ! -d $TOOLCHAIN_PATH ]; then
  echo "Toolchain not found in /opt"
  TOOLCHAIN_ARCHIVE="toolchain-v7-binaries-ubuntu-24.04-amd64.tar.gz"
  if [ ! -f $TOOLCHAIN_ARCHIVE ]; then
    echo "Failed to find toolchain archive: $TOOLCHAIN_ARCHIVE"
    exit 1    
  fi
  sudo tar xzvfm $TOOLCHAIN_ARCHIVE -C /opt
else
  echo "Toolchain found in /opt"
fi



# Check if the 'memgraph' directory exists; if not, clone the repository
if [ ! -d "memgraph" ]; then
  echo "'memgraph' directory not found. Cloning repository..."
  git clone https://github.com/memgraph/memgraph.git
  # check for dependencies
  sudo ./memgraph/environment/os/install_deps.sh install TOOLCHAIN_RUN_DEPS
  sudo ./memgraph/environment/os/install_deps.sh install MEMGRAPH_BUILD_DEPS
  source $TOOLCHAIN_ACTIVATE
  cd memgraph
  ./init
  cmake -B build -S . -D CMAKE_TOOLCHAIN_FILE=toolchain.cmake
  cmake --build ./build --target memgraph -- -j$(nproc)
  cd ..
else
  echo "'memgraph' directory already exists."
  source $TOOLCHAIN_ACTIVATE
fi

# create python env
# if [ ! -d "env" ]; then
#   python3 -m venv env
# fi
# source env/bin/activate

# prepare to build mage
if [ ! -d "mage/dist" ]; then
  echo "Preparing to build mage"
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  export PATH="${HOME}/.cargo/bin:${PATH}"
  python3 -m pip install -r mage/python/requirements.txt --break-system-packages
  python3 -m pip install -r mage/python/tests/requirements.txt  --break-system-packages
  python3 -m pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html --no-cache --break-system-packages
  cd mage
  python3 setup build
  cd $BASE_DIR
else
  echo "Found mage modules"
fi

# Run c++ tests
cd mage/cpp/build
ctest --output-on-failure
cd $BASE_DIR

# Run Python tests
cd mage/python
python3 -m pytest .
cd $BASE_DIR

# Run E2E tests
./memgraph/build/memgraph \
  --query-modules-directory $BASE_DIR/mage/dist \
  --storage-properties-on-edges true \
  --log-level=TRACE \
  --also-log-to-stderr &
cd mage/e2e
python3 -m pytest . -vv -k "not cugraph"
#./test_e2e "not cugraph"
cd $BASE_DIR

# Run E2E correctness (currently expects plugins to be here too)
export PLUGINS_DIR="$BASE_DIR/plugins"
export LOGS_DIR="$BASE_DIR/logs"
mkdir -p $LOGS_DIR
docker run --rm \
    --name testneo4j  \
    -p 7474:7473 \
    -p 7688:7687 \
    -d \
    -v "$HOME/neo4j/plugins:/plugins" \
    --env NEO4J_AUTH=none  \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true  \
    -e NEO4J_PLUGINS='["apoc"]' neo4j:5.10.0
    

sleep 20
# echo "Waiting for Neo4j to start..."
# while ! curl --silent --fail http://localhost:7474; do
#   sleep 1
#   counter=$((counter+1))
#   if [ $counter -gt $timeout ]; then
#     echo "Neo4j failed to start in $timeout seconds"
#     exit 1
#   fi
# done
echo "Neo4j is up and running."

cd mage
python3 test_e2e_correctness.py \
  --memgraph-port 7687 \
  --neo4j-port 7688 \
  --neo4j-container "127.0.0.1"


# clean up
pkill -f "./memgraph/build/memgraph --query-modules-directory /mnt/data-disk-d/mattkjames7/mage/dist --storage-properties-on-edges true --log-level=TRACE"
docker stop testneo4j
