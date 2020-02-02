#!/bin/bash

# modified https://www.tensorflow.org/install/gpu
# cross checked with https://cloud.google.com/compute/docs/gpus/install-drivers-gpu

# install driver for Ubuntu 16.04
export UBUNTU_OS=1604
export CUDA_REPO=cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
 install driver for Ubuntu 18.04 LTS
export UBUNTU_OS=1804
#export CUDA_REPO=cuda-repo-ubuntu1804_10.1.105-1_amd64.deb
export CUDA_REPO=cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu$UBUNTU_OS/x86_64/$CUDA_REPO

sudo dpkg -i $CUDA_REPO
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu$UBUNTU_OS/x86_64/7fa2af80.pub
sudo apt-get update -y
sudo apt-get install cuda

# Unix team maintains a repo of graphics drivers
# https://askubuntu.com/questions/1054954/how-to-install-nvidia-driver-in-ubuntu-18-04
# But this option can't be used because secure boot is enabled
# https://devtalk.nvidia.com/default/topic/1057080/linux/nvidia-smi-has-failed-because-it-couldn-t-communicate-with-the-nvidia-driver/
#sudo add-apt-repository ppa:graphics-drivers -y
#sudo apt-get update -y
## Conda Tensorflow 2.0.0 uses CUDA 10.0 Toolkit, which requires CUDA driver >= 410.48
## https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver
#sudo apt-get install --no-install-recommends -y nvidia-driver-440



# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Check the driver until installed
while ! [[ -x "$(command -v nvidia-smi)" ]];
do
  echo "sleep to check"
  sleep 5s
done
echo "nvidia-smi is installed"

gcloud auth configure-docker
echo "Docker run with GPUs"

# Wait until disk is mounted
export MOUNTPOINT=/mnt/disks/gce-containers-mounts/gce-persistent-disks/big-earth-data
while [[ "$(lsblk -o MOUNTPOINT -nr /dev/sdb)" != $MOUNTPOINT ]]
do
  echo "waiting for disk to be attached to $MOUNTPOINT"
  sleep 5s
  sudo mount /dev/sdb $MOUNTPOINT
done
echo "Mounted disk"

docker run -d --gpus all --log-driver=gcplogs \
-p 8888:8888 \
--volume $MOUNTPOINT:/home/jovyan/work \
us.gcr.io/big-earth-252219/jupyter_tensorflow_notebook \
start-notebook.sh --NotebookApp.password='sha1:53b6a295837d:d096b7b1797ebe5bb5f5ecc355659d760281e343'

echo "started notebook"