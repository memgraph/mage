ARG PY_VERSION_DEFAULT=3.7

FROM debian:buster as base

USER root

ARG MG_VERSION=1.6.1
ARG PY_VERSION_DEFAULT
ENV MG_VERSION ${MG_VERSION}
ENV PY_VERSION ${PY_VERSION_DEFAULT}

#essentials for production/dev
RUN apt-get update && apt-get install -y \
    libcurl4        `memgraph` \
    libpython${PY_VERSION}   `memgraph` \
    libssl1.1       `memgraph` \
    openssl         `memgraph` \
    build-essential `mage-memgraph` \
    cmake           `mage-memgraph` \
    curl            `mage-memgraph` \
    g++             `mage-memgraph` \
    python3         `mage-memgraph` \
    python3-pip     `mage-memgraph` \
    python3-setuptools     `mage-memgraph` \
    uuid-dev        `mage-memgraph` \
    clang           `mage-memgraph` \
    git             `mage-memgraph` \
    --no-install-recommends \
    # Download and install Memgraph
    && curl https://download.memgraph.com/memgraph/v${MG_VERSION}/debian-10/memgraph_${MG_VERSION}-community-1_amd64.deb --output memgraph.deb \
    && dpkg -i memgraph.deb \
    && rm memgraph.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


ENV LD_LIBRARY_PATH /usr/lib/memgraph/query_modules

# Memgraph listens for Bolt Protocol on this port by default.
EXPOSE 7687

FROM base as dev

WORKDIR /mage
COPY . /mage



#MAGE
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && python3 -m  pip install -r /mage/python/requirements.txt \
    && python3 /mage/setup build -p /usr/lib/memgraph/query_modules/


USER memgraph
ENTRYPOINT ["/usr/lib/memgraph/memgraph"]
CMD [""]



FROM base as prod

USER root
ENTRYPOINT []
ARG PY_VERSION_DEFAULT
ENV PY_VERSION ${PY_VERSION_DEFAULT}

#copy modules
COPY --from=dev /usr/lib/memgraph/query_modules/ /usr/lib/memgraph/query_modules/

#copy python build
COPY --from=dev /usr/local/lib/python${PY_VERSION}/ /usr/local/lib/python${PY_VERSION}/


RUN rm -rf /mage \
    && export PATH="/usr/local/lib/python${PY_VERSION}:${PATH}" \
    && apt-get -y --purge autoremove clang git curl python3-pip python3-setuptools cmake build-essential \
    && apt-get clean

USER memgraph
ENTRYPOINT ["/usr/lib/memgraph/memgraph"]
CMD [""]
