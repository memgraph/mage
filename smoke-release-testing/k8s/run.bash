#!/bin/bash -x
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PATH="$(go env GOPATH)/bin:$PATH"
# NOTES:
#   * In the custom values file telemetry was disabled and NodePort was set
#   as the serviceType. The values file was copied from
#   https://github.com/memgraph/helm-charts/blob/main/charts/memgraph-high-availability/values.yaml.
#   * It's critical to run `helm repo update` because otherwise you'll inject
#   latest template that might not be compatible.
#   * `helm uninstall` has to be used otherwise on `kubectl delete pod
#   <pod-name>` the pod gets restarted.
#   * `helm uninstall` is not deleting PVCs, there is the `kubectl delete pvc
#   --all`.
#   * It takes some time to delete all PVs, check with `kubectl get pv`.
#   * If you want more details or helm dry run just append `--debug` of
#   `--dry-run`.
BOLT_SERVER="localhost:10000" # Just tmp value -> each coordinator should have a different value.
# E.g. if kubectl port-foward is used, the configured host values should be passed as `bolt_server`.

# TODO(gitbuda): Move under utils.
load_next_image_into_kind() {
  kind load docker-image $MEMGRAPH_NEXT_DOCKERHUB_IMAGE -n smoke-release-testing
  MEMGRAPH_NEXT_DOCKERHUB_TAG="${MEMGRAPH_NEXT_DOCKERHUB_IMAGE##*:}"
}

setup_coordinator_1_query() {
  echo "ADD COORDINATOR 1 WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\":  \"memgraph-coordinator-1.default.svc.cluster.local:10000\", \"coordinator_server\":  \"memgraph-coordinator-1.default.svc.cluster.local:12000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
}
setup_coordinator_2_query() {
  echo "ADD COORDINATOR 2 WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\":  \"memgraph-coordinator-2.default.svc.cluster.local:10000\", \"coordinator_server\":  \"memgraph-coordinator-2.default.svc.cluster.local:12000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
}
setup_coordinator_3_query() {
  echo "ADD COORDINATOR 3 WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\":  \"memgraph-coordinator-3.default.svc.cluster.local:10000\", \"coordinator_server\":  \"memgraph-coordinator-3.default.svc.cluster.local:12000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
}
setup_replica_0() {
  echo "REGISTER INSTANCE instance_0 WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\": \"memgraph-data-0.default.svc.cluster.local:10000\", \"replication_server\": \"memgraph-data-0.default.svc.cluster.local:20000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
}
setup_replica_1() {
  echo "REGISTER INSTANCE instance_1 WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\": \"memgraph-data-1.default.svc.cluster.local:10000\", \"replication_server\": \"memgraph-data-1.default.svc.cluster.local:20000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
}
setup_main() {
  echo "SET INSTANCE instance_0 TO MAIN;" | $MEMGRAPH_CONSOLE_BINARY --port 17687
}

setup_cluster() {
  kubectl wait --for=condition=Ready pod/memgraph-coordinator-1-0 --timeout=120s
  kubectl wait --for=condition=Ready pod/memgraph-coordinator-2-0 --timeout=120s
  kubectl wait --for=condition=Ready pod/memgraph-coordinator-3-0 --timeout=120s
  kubectl port-forward memgraph-coordinator-1-0 17687:7687 &
  PF_PID=$!
  # TODO(gitbuda): wait + memgraph check.
  sleep 5
  setup_coordinator_1_query
  setup_coordinator_2_query
  setup_coordinator_3_query
  setup_replica_0
  setup_replica_1
  setup_main
  # TODO(gitbuda): make sure this is always executed because the above lines have high chance of failing.
  kill $PF_PID
  wait $PF_PID 2>/dev/null
}

execute_query_against_main() {
  query="$1"
  kubectl wait --for=condition=Ready pod/memgraph-data-0-0 --timeout=120s
  kubectl port-forward memgraph-data-0-0 17687:7687 &
  PF_PID=$!
  wait_for_memgraph localhost 17687
  # TODO: Wait for coordinator OR execute query with timeout (while success loop).
  echo "$query" | $MEMGRAPH_CONSOLE_BINARY --port 17687
  kill $PF_PID
  wait $PF_PID 2>/dev/null || true
}

test_k8s_single() {
  echo "Test k8s single memgraph instance using image: $MEMGRAPH_NEXT_DOCKERHUB_IMAGE"
  load_next_image_into_kind
  helm install memgraph-single-smoke memgraph/memgraph \
    -f "$SCRIPT_DIR/values-single.yaml" \
    --set "image.tag=$MEMGRAPH_NEXT_DOCKERHUB_TAG"
  kubectl wait --for=condition=Ready pod/memgraph-single-smoke-0 --timeout=120s
  kubectl port-forward memgraph-single-smoke-0 17687:7687 &
  PF_PID=$!
  wait_for_memgraph localhost 17687
  echo "CREATE ();" | $MEMGRAPH_CONSOLE_BINARY --port 17687
  echo "MATCH (n) RETURN n;" | $MEMGRAPH_CONSOLE_BINARY --port 17687
  kill $PF_PID
  wait $PF_PID 2>/dev/null || true
  helm uninstall memgraph-single-smoke
}

# TODO(gitbuda): Here we need memgraph/memgraph because that's what's used under the helm chart...
test_k8s_ha() {
  WHICH="$1"
  WHICH_TMP="MEMGRAPH_${WHICH}_DOCKERHUB_IMAGE"
  WHICH_IMAGE="${!WHICH_TMP}"
  MEMGRAPH_DOCKERHUB_TAG="${WHICH_IMAGE##*:}"
  CHART_PATH="${2:-memgraph/memgraph-high-availability}"
  echo "Test k8s HA memgraph cluster using image:"
  echo "  * image: $WHICH_IMAGE"
  echo "  * tag: $MEMGRAPH_DOCKERHUB_TAG"
  echo "  * chart: $CHART_PATH"

  kind load docker-image $WHICH_IMAGE -n smoke-release-testing
  helm install myhadb $CHART_PATH \
    --set env.MEMGRAPH_ENTERPRISE_LICENSE=$MEMGRAPH_ENTERPRISE_LICENSE,env.MEMGRAPH_ORGANIZATION_NAME=$MEMGRAPH_ORGANIZATION_NAME \
    -f "$SCRIPT_DIR/values-ha.yaml" \
    --set "image.tag=$MEMGRAPH_DOCKERHUB_TAG"
  # TODO(gitbuda): On recovery, this is not required -> skip.
  setup_cluster
  # TODO: The sleep here was because it takes some time to init main -> figure that out.
  sleep 5
  # TODO(gitbuda): After recovery, it's uncretain what instance will be MAIN -> figure that out using the query.
  # execute_query_against_main "SHOW REPLICATION ROLE;" OR pull from coordinator "SHOW INSTANCES;"
  execute_query_against_main "SHOW VERSION;"
  execute_query_against_main "CREATE ();"
  execute_query_against_main "MATCH (n) RETURN n;"
  # TODO(gitbuda): flag also for this because we want to debug cluster after something fails.
  helm uninstall myhadb
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  echo "Running $0 directly..."
  # NOTE: Developing workflow: download+load required images and define MEMGRAPH_NEXT_DOCKERHUB_IMAGE.

  # Inject local version of the helm chart because we want to test any local fixes upfront.
  # test_k8s_ha LAST
  # test_k8s_ha NEXT ~/Workspace/code/memgraph/helm-charts/charts/memgraph-high-availability
  test_k8s_ha NEXT

  # kubectl wait --for=condition=Ready pod/memgraph-coordinator-1-0 --timeout=120s
  # kubectl wait --for=condition=Ready pod/memgraph-data-0-0 --timeout=120s
  # kubectl port-forward memgraph-coordinator-1-0 17687:7687 &
  # PF_PID=$!
  # sleep 2
  # setup_main
  # kill $PF_PID
  # wait $PF_PID 2>/dev/null

  # execute_query_against_main "CREATE ();"
  # execute_query_against_main "MATCH (n) RETURN n;"
fi
