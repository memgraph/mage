#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../utils.bash"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  # NOTE: In the custom values file telemetry was disabled and NodePort was set
  # as the serviceType. The values file was copied from
  # https://github.com/memgraph/helm-charts/blob/main/charts/memgraph-high-availability/values.yaml.
  # NOTE: It's critical to run `helm repo update` because otherwise you'll
  # inject latest template that might not be compatible.
  helm install myhadb memgraph/memgraph-high-availability \
    --set env.MEMGRAPH_ENTERPRISE_LICENSE=$MEMGRAPH_ENTERPRISE_LICENSE,env.MEMGRAPH_ORGANIZATION_NAME=$MEMGRAPH_ORGANIZATION_NAME \
    -f $SCRIPT_DIR/ha-values.yaml
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
  # NOTE: helm uninstall is not deleting PVCs
  # kubectl delete pvc --all
  # NOTE: it take some time to delete all PVs, check with
  # kubectl get pv
  #### DEBUGGING
  # kubectl exec -it memgraph-coordinator-3-0 -- bash
fi
