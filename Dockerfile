FROM debian:buster as production
USER root

RUN apt-get update && apt-get install -y \
    build-essential `mage` \
    cmake           `mage` \
    curl            `mage-memgraph` \
    g++             `mage` \
    git             `mage` \
    libcurl4        `memgraph` \
    libpython3.7    `memgraph` \
    libssl1.1       `memgraph` \
    openssl         `memgraph` \
    python3         `mage-memgraph` \
    python3-pip     `mage-memgraph` \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#Memgraph
RUN curl https://download.memgraph.com/memgraph/v1.6.1/debian-10/memgraph_1.6.1-community-1_amd64.deb --output memgraph.deb \
    && dpkg -i memgraph.deb \
    && rm memgraph.deb

#MAGE
RUN apt-get update && apt-get install -y clang --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && git clone https://github.com/memgraph/mage.git \
    && cd /mage \
    && python3 /mage/build \
    && cp -r /mage/dist/* /usr/lib/memgraph/query_modules/ \
    && python3 -m  pip install -r /mage/python/requirements.txt \
    && rm -rf /mage \
    && apt-get -y --purge autoremove clang git curl python3-pip cmake \
    && apt-get clean \
    && rustup self uninstall -y &> dev/null


ENV LD_LIBRARY_PATH /usr/lib/memgraph/query_modules

# Memgraph listens for Bolt Protocol on this port by default.
EXPOSE 7687

WORKDIR /usr/lib/memgraph/query_modules

CMD ["runuser","-l","memgraph", "-c", "/usr/lib/memgraph/memgraph"]

#Development
FROM production as development
RUN apt-get update && apt-get install -y  \
    clang  \
    python3-pip  \
    cmake  \
    curl  \
    --no-install-recommends \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y