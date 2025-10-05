#!/bin/bash

echo "Running container setup..."

rm -f /usr/lib/memgraph/query_modules/embed_worker/embed_worker.py
ln -s /app/embed_worker.py /usr/lib/memgraph/query_modules/embed_worker/embed_worker.py

rm -f /usr/lib/memgraph/query_modules/embeddings.py
ln -s /app/embeddings.py /usr/lib/memgraph/query_modules/embeddings.py
