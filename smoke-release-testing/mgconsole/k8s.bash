#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"

# TODO(gitbuda): SAVE https://kubernetes.io/docs/reference/kubectl/quick-reference/
# TODO(gitbuda): https://mcvidanagama.medium.com/set-up-a-multi-node-kubernetes-cluster-locally-using-kind-eafd46dd63e5

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: In the custom values file affinity and telemetry were disabled.
  helm install myhadb memgraph/memgraph-high-availability \
    --set env.MEMGRAPH_ENTERPRISE_LICENSE=$MEMGRAPH_ENTERPRISE_LICENSE,env.MEMGRAPH_ORGANIZATION_NAME=$MEMGRAPH_ORGANIZATION_NAME \
    -f $SCRIPT_DIR/config/k8s-ha-values.yaml
  sleep 1000
  # helm list
  # kubectl get pods
  # kubectl get pod $pod_name -o yaml
  # kubectl logs $pod_name
  # kubectl describe pods $pod_name
  # kubectl cluster-info dump
  ## https://komodor.com/learn/kubectl-port-forwarding-how-it-works-use-cases-examples
  # kubectl port-forward $pod_name 17687:7687
  helm uninstall myhadb
fi
