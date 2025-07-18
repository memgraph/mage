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

setup_coordinator() {
  local i=$1
  echo "ADD COORDINATOR $i WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\":  \"memgraph-coordinator-$i.default.svc.cluster.local:10000\", \"coordinator_server\":  \"memgraph-coordinator-$i.default.svc.cluster.local:12000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
  echo "coordinator $i DONE"
}
setup_replica() {
  local i=$1
  echo "REGISTER INSTANCE instance_$i WITH CONFIG {\"bolt_server\": \"$BOLT_SERVER\", \"management_server\": \"memgraph-data-$i.default.svc.cluster.local:10000\", \"replication_server\": \"memgraph-data-$i.default.svc.cluster.local:20000\"};" | $MEMGRAPH_CONSOLE_BINARY --port 17687
  echo "replica $i DONE"
}
setup_main() {
  local i=$1
  echo "SET INSTANCE instance_$i TO MAIN;" | $MEMGRAPH_CONSOLE_BINARY --port 17687
  echo "main DONE"
}

setup_cluster() {
  kubectl wait --for=condition=Ready pod -l role=coordinator --timeout=120s
  kubectl wait --for=condition=Ready pod -l role=data --timeout=120s

  # TODO(gitbuda): The reason for the abstracted code is also that sometimes port-foward fails (again some sync issue).
  # TODO(gitbuda): An attempt to make the code nicer but it still doesn't work -> not all gets executed + there is an infinitely loop.
  with_kubectl_portforward memgraph-coordinator-1-0 17687:7687 -- \
    'wait_for_memgraph_coordinator localhost 17687' \
    'setup_coordinator 1' \
    'setup_coordinator 2' \
    'setup_coordinator 3' \
    'setup_replica 0' \
    'setup_main 0'
}

execute_query_against_main() {
  query="$1"

  # Derive what's the main instance (it's not deterministic where is MAIN after recovery).
  # kubectl wait --for=condition=Ready pod/memgraph-coordinator-1-0 --timeout=120s
  # kubectl port-forward memgraph-coordinator-1-0 17687:7687 &
  # PF_PID=$!
  # kill $PF_PID
  # wait $PF_PID 2>/dev/null || true

  # kubectl wait --for=condition=Ready pod/${main_instance}-0 --timeout=120s
  # kubectl port-forward ${main_instance}-0 17687:7687 &
  # PF_PID=$!
  # kill $PF_PID
  # wait $PF_PID 2>/dev/null || true

  with_kubectl_portforward memgraph-coordinator-1-0 17687:7687 -- \
    "wait_for_memgraph_coordinator localhost 17687" \
    "export MAIN_INSTANCE=\$(echo \"SHOW INSTANCES;\" | $MEMGRAPH_CONSOLE_BINARY --port 17687 --output-format=csv | python3 $SCRIPT_DIR/../reader.py get_main_parser)" \
    "echo \"NOTE: MAIN instance is \$MAIN_INSTANCE\""
  # TODO(gitbuda): MAIN_INSTANCE variable here is lost -> FIX
  with_kubectl_portforward $MAIN_INSTANCE 17687:7687 -- \
    "wait_for_memgraph localhost 17687" \
    "echo \"$query\" | $MEMGRAPH_CONSOLE_BINARY --port 17687"
}

test_k8s_single() {
  # TODO(gitbuda): Refactor test_k8s_single to also use with_kubectl_portforward
  echo "Test k8s single memgraph instance using image: $MEMGRAPH_NEXT_DOCKERHUB_IMAGE"
  kind load docker-image $MEMGRAPH_NEXT_DOCKERHUB_IMAGE -n smoke-release-testing
  MEMGRAPH_NEXT_DOCKERHUB_TAG="${MEMGRAPH_NEXT_DOCKERHUB_IMAGE##*:}"
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

test_k8s_help() {
  echo "usage: test_k8s_ha LAST|NEXT [-p|--chart-path PATH] [-s|--skip-cluster-setup] [-c|--skip-cleanup] [-h|--help]"
  exit 1
}

test_k8s_ha() {
  if [ "$#" -lt 1 ]; then
    test_k8s_help
  fi
  WHICH="$1"
  WHICH_TMP="MEMGRAPH_${WHICH}_DOCKERHUB_IMAGE"
  WHICH_IMAGE="${!WHICH_TMP}"
  MEMGRAPH_DOCKERHUB_TAG="${WHICH_IMAGE##*:}"
  shift
  CHART_PATH="memgraph/memgraph-high-availability"
  SKIP_CLUSTER_SETUP=false
  SKIP_CLEANUP=false
  SKIP_HELM_UNINSTALL=false
  while true; do
    case $1 in
      -p|--chart-path)          CHART_PATH="$2";          shift 2 ;;
      -s|--skip-cluster-setup)  SKIP_CLUSTER_SETUP=true;  shift ;;
      -u|--skip-helm-uninstall) SKIP_HELM_UNINSTALL=true; shift ;;
      -c|--skip-cleanup)        SKIP_CLEANUP=true;        shift ;;
      -h|--help)                test_k8s_help;            ;;
      *)                        shift;                    break ;;
    esac
  done
  echo "Test k8s HA memgraph cluster using image:"
  echo "  * image: $WHICH_IMAGE"
  echo "  * tag: $MEMGRAPH_DOCKERHUB_TAG"
  echo "  * chart: $CHART_PATH"
  echo "  * skip cluster setup: $SKIP_CLUSTER_SETUP"
  echo "  * skip cleanup: $SKIP_CLEANUP"

  kind load docker-image $WHICH_IMAGE -n smoke-release-testing
  helm install myhadb $CHART_PATH \
    --set env.MEMGRAPH_ENTERPRISE_LICENSE=$MEMGRAPH_ENTERPRISE_LICENSE,env.MEMGRAPH_ORGANIZATION_NAME=$MEMGRAPH_ORGANIZATION_NAME \
    -f "$SCRIPT_DIR/values-ha.yaml" \
    --set "image.tag=$MEMGRAPH_DOCKERHUB_TAG"
  if [ "$SKIP_CLUSTER_SETUP" = false ]; then
    setup_cluster
  fi
  execute_query_against_main "SHOW VERSION;"
  execute_query_against_main "CREATE ();"
  execute_query_against_main "MATCH (n) RETURN n;"
  if [ "$SKIP_HELM_UNINSTALL" = false ]; then
    helm uninstall myhadb
  fi
  if [ "$SKIP_CLEANUP" = false ]; then
    kubectl delete pvc --all
  fi
}

call_me() {
  echo "x"
}
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  echo "Running $0 directly..."
  # NOTE: Developing workflow: download+load required images and define MEMGRAPH_NEXT_DOCKERHUB_IMAGE.

  # test_k8s_ha LAST -c # Skip cleanup because we want to recover from existing PVCs.
  # test_k8s_ha NEXT -s # Skip cluster setup because that should be recovered from PVCs.

  # How to inject local version of the helm chart because we want to test any local fixes upfront.
  # test_k8s_ha NEXT ~/Workspace/code/memgraph/helm-charts/charts/memgraph-high-availability

  # TODO(gitbuda): At the moment, it can happen that exit code is 0 while there is a failure -> FIX.
  # test_k8s_ha NEXT -u -c

  with_kubectl_portforward memgraph-coordinator-1-0 17687:7687 -- \
    'export EXP_VAR="foo"'
    "echo \"MG_MAIN=\"data0\" > $SCRIPT_DIR/mg_main.out"
  source $SCRIPT_DIR/mg_main.out
  echo "$MG_MAIN"
fi
