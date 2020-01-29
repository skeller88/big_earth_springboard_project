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
docker run -d --gpus all --log-driver=gcplogs \
-p 8888:8888 \
--volume /mnt/disks/gce-containers-mounts/gce-persistent-disks/big-earth-data:/home/jovyan/work \
us.gcr.io/big-earth-252219/jupyter_tensorflow_notebook \
start-notebook.sh --NotebookApp.password='sha1:53b6a295837d:d096b7b1797ebe5bb5f5ecc355659d760281e343'

echo "started notebook"