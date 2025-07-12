#!/bin/bash
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

# TODO(gitbuda): Move under utils.
load_next_image_into_kind() {
  kind load docker-image $MEMGRAPH_NEXT_DOCKERHUB_IMAGE -n smoke-release-testing
  MEMGRAPH_NEXT_DOCKERHUB_TAG="${MEMGRAPH_NEXT_DOCKERHUB_IMAGE##*:}"
}

# TODO(gitbuda): Setup memgraph HA cluster.

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
  echo "Test k8s HA memgraph cluster using image: $WHICH_IMAGE"
  kind load docker-image $WHICH_IMAGE -n smoke-release-testing
  MEMGRAPH_DOCKERHUB_TAG="${WHICH_IMAGE##*:}"
  echo $MEMGRAPH_DOCKERHUB_TAG
  helm install myhadb memgraph/memgraph-high-availability \
    --set env.MEMGRAPH_ENTERPRISE_LICENSE=$MEMGRAPH_ENTERPRISE_LICENSE,env.MEMGRAPH_ORGANIZATION_NAME=$MEMGRAPH_ORGANIZATION_NAME \
    -f "$SCRIPT_DIR/values-ha.yaml" \
    --set "image.tag=$MEMGRAPH_DOCKERHUB_TAG"
  # TODO(gitbuda): Setup cluster commands + routing + test query.
  helm uninstall myhadb
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: Developing workflow: download+load required images and define MEMGRAPH_NEXT_DOCKERHUB_IMAGE.
  test_k8s_ha
fi
