#!/bin/bash

# Check the driver until installed
while ! [[ -x "$(command -v nvidia-smi)" ]];
do
  echo "sleep to check"
  sleep 5s
done
echo "nvidia-smi is installed"

gcloud auth configure-docker
echo "Docker run with GPUs"

sudo mount /dev/sdb /mnt/disks/gce-containers-mounts/gce-persistent-disks/big-earth-data
echo "Mounted disk"

docker run -d --gpus all --log-driver=gcplogs \
-p 8888:8888 \
--volume /mnt/disks/gce-containers-mounts/gce-persistent-disks/big-earth-data:/home/jovyan/work \
us.gcr.io/big-earth-252219/papermill_jupyter_tensorflow_notebook

echo "started notebook"