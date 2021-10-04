FROM memgraph/memgraph:latest AS memgraph-mage

FROM memgraph-mage
USER root

RUN apt-get update && \
    apt-get --yes install curl git cmake g++ clang python3-dev uuid-dev python3-setuptools && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set mage as working directory
WORKDIR /mage
COPY . /mage

# Build all necessary modules
RUN python3 build
RUN cp -r /mage/dist/* /usr/lib/memgraph/query_modules/

# It's required to install python3 because auth module scripts are going to be
# written in python3.
RUN python3 -m  pip install -r /mage/python/requirements.txt

USER memgraph
ENV LD_LIBRARY_PATH /usr/lib/memgraph/query_modules
# Memgraph listens for Bolt Protocol on this port by default.
EXPOSE 7687
# Snapshots and logging volumes
VOLUME /var/log/memgraph
VOLUME /var/lib/memgraph
VOLUME /etc/memgraph

WORKDIR /usr/lib/memgraph/query_modules
