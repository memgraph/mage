#!/bin/bash

################################################################################
# Use this script within the `prod` container to convert it to a dev container #
# Optionally skip toolchain download with --no-toolchain                       #
# run me as root!                                                              #
################################################################################

# install dependencies
apt-get update 
apt-get install -y libpython${PY_VERSION:-$(python3 --version | sed 's/Python //')} \
  libcurl4 libssl-dev openssl build-essential cmake curl g++ python3  \
  python3-pip python3-setuptools python3-dev clang git unixodbc-dev \
  libboost-all-dev uuid-dev gdb procps libc6-dbg libxmlsec1-dev xmlsec1 \
  --no-install-recommends 


# Function to detect architecture and set the appropriate toolchain URL
get_toolchain_url() {
  arch=$(uname -m)
  case "$arch" in
    x86_64)
      # AMD64 architecture
      echo "https://s3-eu-west-1.amazonaws.com/deps.memgraph.io/toolchain-v6/toolchain-v6-binaries-ubuntu-24.04-amd64.tar.gz"
      ;;
    aarch64)
      # ARM64 architecture
      echo "https://s3-eu-west-1.amazonaws.com/deps.memgraph.io/toolchain-v6/toolchain-v6-binaries-ubuntu-24.04-arm64.tar.gz"
      ;;
    *)
      echo "Unsupported architecture: $arch" >&2
      exit 1
      ;;
  esac
}


TOOLCHAIN_URL=$(get_toolchain_url)
echo "Downloading toolchain from: $TOOLCHAIN_URL"
# Download the toolchain using curl
curl -L "$TOOLCHAIN_URL" -o /toolchain.tar.gz
tar xzvfm /toolchain.tar.gz -C /opt


echo "Cloning MAGE repo commit/tag: $MAGE_COMMIT"
cd /root
git clone https://github.com/memgraph/mage.git --recurse-submodules
cd /root/mage
git checkout $MAGE_COMMIT
cd /

echo "Copying repo files to /mage"
# Copy files without overwriting existing ones
cp -r --update=none /root/mage/. /mage/

# Change ownership of everything in /mage to memgraph
chown -R memgraph: /mage/

# remove git repo from `/root`
rm -rf /root/mage

# install Rust as `memgraph` user
su - memgraph -c 'source /opt/toolchain-v6/activate && curl https://sh.rustup.rs -sSf | sh -s -- -y && echo "export PATH=\$HOME/.cargo/bin:\$PATH" >> $HOME/.bashrc'

# build everything again (because it isn't copied into the prod image)
su - memgraph -c 'cd /mage && python3 /mage/setup build --cpp-build-flags CMAKE_BUILD_TYPE=$BUILD_TYPE'

echo "Done!"
