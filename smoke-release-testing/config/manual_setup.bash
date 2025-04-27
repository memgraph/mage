#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"
BOLT_SERVER="localhost:10000" # Just tmp value -> each coordinator should have a different value.

# NOTE: This script is flaky because it's unknown who will actually be the
# leader. How to know who will be the leader or ask who is the leader?

kubectl port-forward memgraph-coordinator-1-0 17687:7687 &
PF_PID=$!
sleep 2
echo "ADD COORDINATOR 1 WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\":  \"memgraph-coordinator-1.default.svc.cluster.local:10000\", \"coordinator_server\":  \"memgraph-coordinator-1.default.svc.cluster.local:12000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
kill $PF_PID
wait $PF_PID 2>/dev/null

kubectl port-forward memgraph-coordinator-2-0 17687:7687 &
PF_PID=$!
sleep 2
echo "ADD COORDINATOR 2 WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\":  \"memgraph-coordinator-2.default.svc.cluster.local:10000\", \"coordinator_server\":  \"memgraph-coordinator-2.default.svc.cluster.local:12000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
kill $PF_PID
wait $PF_PID 2>/dev/null

kubectl port-forward memgraph-coordinator-3-0 17687:7687 &
PF_PID=$!
sleep 2
echo "ADD COORDINATOR 3 WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\":  \"memgraph-coordinator-3.default.svc.cluster.local:10000\", \"coordinator_server\":  \"memgraph-coordinator-3.default.svc.cluster.local:12000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
kill $PF_PID
wait $PF_PID 2>/dev/null

kubectl port-forward memgraph-coordinator-1-0 17687:7687 &
PF_PID=$!
sleep 2
echo "REGISTER INSTANCE instance_0 WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\": \"memgraph-data-0.default.svc.cluster.local:10000\", \"replication_server\": \"memgraph-data-0.default.svc.cluster.local:20000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
echo "REGISTER INSTANCE instance_1 WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\": \"memgraph-data-1.default.svc.cluster.local:10000\", \"replication_server\": \"memgraph-data-1.default.svc.cluster.local:20000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
echo "SET INSTANCE instance_1 TO MAIN;" | $MEMGRAPH_CONSOLE_BINARY --port 17687
kill $PF_PID
wait $PF_PID 2>/dev/null
