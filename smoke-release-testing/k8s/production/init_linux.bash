#!/bin/bash -e

# Sources:
#   * https://medium.com/@subhampradhan966/kubeadm-setup-for-ubuntu-24-04-lts-f6a5fc67f0df
#   * https://chatgpt.com/share/6812882c-f450-800a-b13c-04eefccc0880

sudo swapoff -a
# sudo sed -i '/ swap / s/^/#/' /etc/fstab

sudo apt update -y
sudo apt install -y apt-transport-https ca-certificates curl

sudo apt install -y containerd
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml > /dev/null
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl restart containerd

curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.30/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.30/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt update
sudo apt install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

## master

sudo kubeadm init --apiserver-advertise-address=100.x.x.x --pod-network-cidr=10.244.0.0/16
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
# NOTE: The latest version of flannel doesn't work.
# kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
curl -O https://raw.githubusercontent.com/flannel-io/flannel/v0.25.6/Documentation/kube-flannel.yml
kubectl apply -f kube-flannel.yml
kubeadm token create --print-join-command

## worker

sudo sysctl -w net.ipv4.ip_forward=1
sudo kubectl join ...
# TODO(gitbuda): Config on the worker is also wrong -> local ip address is used.
