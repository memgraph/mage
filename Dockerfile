FROM debian:buster as base
USER root

#essentials for production/dev
RUN apt-get update && apt-get install -y \
    build-essential `mage-memgraph` \
    cmake           `mage-memgraph` \
    curl            `mage-memgraph` \
    g++             `mage-memgraph` \
    libcurl4        `memgraph` \
    libpython3.7    `memgraph` \
    libssl1.1       `memgraph` \
    openssl         `memgraph` \
    python3         `mage-memgraph` \
    python3-pip     `mage-memgraph` \
    uuid-dev        `mage-memgraph` \
    clang           `mage-memgraph` \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


#Memgraph
RUN curl https://download.memgraph.com/memgraph/v1.6.1/debian-10/memgraph_1.6.1-community-1_amd64.deb --output memgraph.deb \
    && dpkg -i memgraph.deb \
    && rm memgraph.deb

ENV LD_LIBRARY_PATH /usr/lib/memgraph/query_modules

# Memgraph listens for Bolt Protocol on this port by default.
EXPOSE 7687


FROM base as production

#MAGE
RUN apt-get update && apt-get install -y git --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && git clone https://github.com/memgraph/mage.git \
    && cd /mage \
    && python3 /mage/build \
    && python3 -m  pip install -r /mage/python/requirements.txt \
    && cp -r /mage/dist/* /usr/lib/memgraph/query_modules/ \
    && rm -rf /mage \
    && apt-get -y --purge autoremove clang git curl python3-pip cmake build-essential \
    && rm -rf /root/.rustup/toolchains \
    && apt-get clean


CMD ["runuser","-l","memgraph", "-c", "/usr/lib/memgraph/memgraph"]

#Development
FROM base as development

WORKDIR /mage
COPY ./ /mage/

#rust missing still
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && python3 /mage/build \
    && python3 -m  pip install -r /mage/python/requirements.txt \
    && cp -r /mage/dist/* /usr/lib/memgraph/query_modules/

CMD ["runuser","-l","memgraph", "-c", "/usr/lib/memgraph/memgraph"]