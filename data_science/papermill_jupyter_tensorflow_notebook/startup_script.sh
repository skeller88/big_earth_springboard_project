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

git clone https://github.com/GoogleCloudPlatform/tensorflow-inference-tensorrt5-t4-gpu.git
cd tensorflow-inference-tensorrt5-t4-gpu/metrics_reporting
pip install -r ./requirements.txt
sudo cp report_gpu_metrics.py /root/
cat <<-EOH > /lib/systemd/system/gpu_utilization_agent.service
[Unit]
Description=GPU Utilization Metric Agent
[Service]
PIDFile=/run/gpu_agent.pid
ExecStart=/bin/bash --login -c '/usr/bin/python /root/report_gpu_metrics.py'
User=root
Group=root
WorkingDirectory=/
Restart=always
[Install]
WantedBy=multi-user.target
EOH

docker run -d --gpus all --log-driver=gcplogs \
-p 8888:8888 \
--volume /mnt/disks/gce-containers-mounts/gce-persistent-disks/big-earth-data:/home/jovyan/work \
us.gcr.io/big-earth-252219/papermill_jupyter_tensorflow_notebook

echo "started notebook"