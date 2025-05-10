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
# TODO(gitbuda): kind requires docker to be installed.
if ! kubectl cluster-info --context kind-experiment; then
  kind create cluster --name experiment
fi
kubectl get all -A

helm repo add memgraph https://memgraph.github.io/helm-charts
helm repo update
helm repo list
# helm install my-release memgraph/memgraph # TODO: Fails if it's already there -> figure out how to skip.

# NOTE: Downloading and compiling that last mgconsole.
# rm -rf $SCRIPT_DIR/mgconsole.build # To download and rebuild everything.
if [ ! -d "$SCRIPT_DIR/mgconsole.build" ]; then
  git clone git@github.com:memgraph/mgconsole.git "$SCRIPT_DIR/mgconsole.build"
fi
MG_CONSOLE_TAG="master"
MG_CONSOLE_BINARY="$SCRIPT_DIR/mgconsole.build/build/src/mgconsole"
if [ ! -f "$MG_CONSOLE_BINARY" ]; then
  cd "$SCRIPT_DIR/mgconsole.build"
  git checkout $MG_CONSOLE_TAG
  mkdir -p build && cd build
  cmake -DCMAKE_RELEASE_TYPE=Release ..
  make -j8
fi
if [ -x "$MG_CONSOLE_BINARY" ]; then
  echo "mgconsole available"
else
  echo "failed to build mgconsole"
fi

rm $SCRIPT_DIR/get_helm.sh || true
