#!/bin/bash
exec /usr/lib/memgraph/memgraph --telemetry-enabled=False --log-level=TRACE 2>&1 | tee /logs/memgraph.log
