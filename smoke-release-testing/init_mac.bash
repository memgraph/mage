#!/bin/bash -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/utils.bash"

if [ ! -x "$(command -v go)" ]; then
  brew install go
fi
go version

if [ ! -f "$(go env GOPATH)/bin/kind" ]; then
  go install sigs.k8s.io/kind@v0.24.0
  echo "kind installed under $(go env GOPATH)/bin"
fi
export PATH="$(go env GOPATH)/bin:$PATH"
kind --version

if [ ! -f "/usr/local/bin/kubectl" ]; then
  echo "TODO: install kubectl"
  exit 1
fi
kubectl version --client

if [ ! -f "/usr/local/bin/helm" ]; then
  curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
  chmod 700 get_helm.sh
  ./get_helm.sh
fi
helm version

# kubectl config get-clusters
if ! kubectl cluster-info --context kind-experiment; then
  kind create cluster --name experiment
fi
kubectl get all -A

helm repo add memgraph https://memgraph.github.io/helm-charts
helm update
helm list
helm my-release memgraph/memgraph

rm $SCRIPT_DIR/get_helm.sh || true
