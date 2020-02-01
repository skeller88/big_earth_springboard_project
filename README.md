# Local Python Environment
```bash
export PROJECT_DIR=<project-dir>
cd $PROJECT_DIR
python3 -m venv venv
pip install -r requirements.txt

# For use in building python dependencies to submit to spark
pip install -U --pre pybuilder

# python3 -m pip install --user --upgrade pip

# Jupyter should already be installed
python -m ipykernel install --user --name=big_earth_springboard_project
```

# Google Cloud Environment
Create auth key file and download to local machine.
 
```bash 
export KEY_FILE=[your-key-file]
gcloud auth activate-service-account --key-file=$KEY_FILE
gcloud auth configure-docker

export PROJECT_ID=[your-project-id]
export HOSTNAME=us.gcr.io
export BASE_IMAGE_NAME=$HOSTNAME/$PROJECT_ID
```

# Data exploration and preparation with Dask/Spark
## Prototype locally
```
docker pull jupyter/tensorflow-notebook
docker run -it --rm -p 8888:8888 --volume ~:/home/jovyan/work jupyter/tensorflow-notebook
```

Navigate to the `data_engineering/data_aggregator` folder for prototype
notebooks.

## Run ETL image locally and deploy to google cloud
export FILEDIR=data_engineering/archive_etler
export IMAGE_NAME=$BASE_IMAGE_NAME/archive_etler
docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile .
docker push $IMAGE_NAME
docker run -it --rm -p 8889:8889 \
--volume ~:/big-earth-data \
--env-file $FILEDIR/env.list $IMAGE_NAME

## Run on google cloud
```
# First time
gcloud compute instances create-with-container archive-etler \
        --zone=us-west1-b \
        --container-env-file=$FILEDIR/env.list \
        --container-image=$IMAGE_NAME \
        --container-mount-disk=name=big-earth-data,mount-path=/big-earth-data,mode=rw \
        --container-restart-policy=never \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-4 \
        --metadata-from-file=startup-script=startup_script.sh \
        --boot-disk-size=10GB \
        --create-disk=name=big-earth-data,auto-delete=no,mode=rw,size=200GB,type=pd-ssd,device-name=big-earth-data

# Subsequent times, attach disk
`--disk=name=big-earth-data,auto-delete=no,mode=rw,device-name=big-earth-data`

# start and stop
gcloud compute instances stop archive-etler
gcloud compute instances start archive-etler

# ssh to instance and unmount disk
sudo umount /dev/disk/by-id/google-big-earth-data

# stop instance and detach the disk when done with ETL
gcloud compute instances stop archive-etler
gcloud compute instances detach-disk archive-etler --disk=big-earth-data

# Reattach disk
gcloud compute instances attach-disk archive-etler \
    --disk=big-earth-data \
    --device-name=big-earth-data \
    --mode=rw \
    --zone=us-west1-b
```

gsutil cp -R gs://big_earth/png_image_files gs://big_earth_us_central_1


# Model training
## Prototype with Jupyter notebook
```
export FILEDIR=data_science/jupyter_tensorflow_notebook
export IMAGE_NAME=$BASE_IMAGE_NAME/jupyter_tensorflow_notebook
docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile  .
docker push $IMAGE_NAME
docker run -it --rm -p 8888:8888 --volume ~:/home/jovyan/work $IMAGE_NAME

gcloud compute addresses create jupyter-tensorflow-notebook --region us-west1
gcloud compute addresses list

# Copy address from above output
export IP_ADDRESS=[ip-address]
export DISK_NAME=big-earth-data

```

## Create GCP instance from Google image family
```
# scopes needed are pub/sub, service control, service management, container registry,
# stackdriver logging/trace/monitoring, storage
# Full names: --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/pubsub,https://www.googleapis.com/auth/logging.admin,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/source.read_only \
export IMAGE_FAMILY="tf2-latest-gpu"
gcloud compute instances create jupyter-tensorflow-notebook \
        --zone=us-west1-b \
        --accelerator=count=1,type=nvidia-tesla-v100 \
        --can-ip-forward \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --scopes=cloud-platform,cloud-source-repos-ro,compute-rw,datastore,default,storage-rw \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-4 \
        --boot-disk-size=50GB \
        --metadata=enable-oslogin=TRUE,install-nvidia-driver=True \
        --metadata-from-file=startup-script=$FILEDIR/startup_script.sh \
        --disk=name=$DISK_NAME,auto-delete=no,mode=rw,device-name=$DISK_NAME

# SSH to instance
# password is
export DISK_NAME=big-earth-data
export JUPYTER_USER=jovyan
export JUPYTER_USER_DIR = /mnt/disks/gce-containers-mounts/gce-persistent-disks/$DISK_NAME/$JUPYTER_USER
sudo useradd $JUPYTER_USER -g users

sudo mkdir $JUPYTER_USER_DIR
sudo chwon $JUPYTER_USER:users $JUPYTER_USER_DIR
docker run -d -p 8888:8888 \
--volume /mnt/disks/gce-containers-mounts/gce-persistent-disks/$DISK_NAME:/home/jovyan/work \
us.gcr.io/big-earth-252219/jupyter_tensorflow_notebook \
start-notebook.sh --NotebookApp.password='sha1:53b6a295837d:d096b7b1797ebe5bb5f5ecc355659d760281e343'

# --user root -e NB_USER=$JUPYTER_USER -e NB_GROUP=users \
# time to make party

# if the disk is not found, confirm the disk is attached to the instance
lsblk
sudo mount /dev/sdb /mnt/disks/gce-containers-mounts/gce-persistent-disks/big-earth-data

# Stop and start
gcloud compute instances stop jupyter-tensorflow-notebook
gcloud compute instances start jupyter-tensorflow-notebook

# Delete
gcloud compute instances delete jupyter-tensorflow-notebook
```

## Create GCP instance from Google image family to run Papermill training notebook
```
export FILEDIR=data_science/papermill_jupyter_tensorflow_notebook
export IMAGE_NAME=$BASE_IMAGE_NAME/papermill_jupyter_tensorflow_notebook
docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile  .
docker push $IMAGE_NAME
docker run -it --rm -p 8888:8888 --volume ~:/home/jovyan/work $IMAGE_NAME

# scopes needed are pub/sub, service control, service management, container registry,
# stackdriver logging/trace/monitoring, storage
# Full names: --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/pubsub,https://www.googleapis.com/auth/logging.admin,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/source.read_only \
export IMAGE_FAMILY="tf2-latest-gpu"
gcloud compute instances create papermill-jupyter-tensorflow-notebook \
        --zone=us-west1-b \
        --accelerator=count=1,type=nvidia-tesla-v100 \
        --can-ip-forward \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --scopes=cloud-platform,cloud-source-repos-ro,compute-rw,datastore,default,storage-rw \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-4 \
        --boot-disk-size=50GB \
        --metadata=enable-oslogin=TRUE,install-nvidia-driver=True \
        --metadata-from-file=startup-script=$FILEDIR/startup_script.sh \
        --disk=name=$DISK_NAME,auto-delete=no,mode=rw,device-name=$DISK_NAME
```

## Create GCP instance from Docker image
TODO
- fix file system permissions error
- fix missing tensorflow .so files
```
# scopes needed are pub/sub, service control, service management, container registry,
# stackdriver logging/trace/monitoring, storage
# Full names: --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/pubsub,https://www.googleapis.com/auth/logging.admin,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/source.read_only \
gcloud compute instances create-with-container jupyter-tensorflow-notebook \
        --address=$IP_ADDRESS \
        --zone=us-west1-b \
        --accelerator=count=1,type=nvidia-tesla-v100 \
        --can-ip-forward \
        --container-image=$IMAGE_NAME \
        --container-mount-disk=name=$DISK_NAME,mount-path=/$DISK_NAME,mode=rw \
        --scopes=cloud-platform,cloud-source-repos-ro,compute-rw,datastore,default,storage-rw \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-4 \
        --boot-disk-size=50GB \
        --metadata enable-oslogin=TRUE \
        --disk=name=$DISK_NAME,auto-delete=no,mode=rw,device-name=$DISK_NAME

# SSH to instance
# password is
Take me to your river succulent
export DISK_NAME=big-earth-data
sudo chmod -R 777 /mnt/disks/gce-containers-mounts/gce-persistent-disks/$DISK_NAME
docker run -p 8888:8888 \
--user root -e NB_GROUP=users \
--volume /mnt/disks/gce-containers-mounts/gce-persistent-disks/$DISK_NAME:/home/jovyan/work \
us.gcr.io/big-earth-252219/jupyter_tensorflow_notebook \
start-notebook.sh --NotebookApp.password='sha1:3f5b37350f9d:716aaf131f345da3352ded1f952e6b36bea8add8'



# Stop and start
gcloud compute instances stop jupyter-tensorflow-notebook
gcloud compute instances start jupyter-tensorflow-notebook

# Delete
gcloud compute instances delete jupyter-tensorflow-notebook
```

## Train
```
export FILEDIR=data_science/model_trainer
export IMAGE_NAME=$BASE_IMAGE_NAME/model_trainer
docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile --build-arg filedir=$FILEDIR .
docker push $IMAGE_NAME
docker run -it --rm -p 8888:8888 --volume ~:/big-earth-data $IMAGE_NAME

gcloud compute instances create-with-container model-trainer \
        --zone=us-west1-b \
        --accelerator=count=1,type=nvidia-tesla-v100 \
        --can-ip-forward \
        --container-image=$IMAGE_NAME \
        --container-mount-disk=name=big-earth-data,mount-path=/big-earth-data,mode=rw \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-4 \
        --boot-disk-size=10GB \
        --disk=name=big-earth-data,mode=rw,auto-delete=no,device-name=big-earth-data
```

# Scratch work

## Local prototype with Spark notebook
```bash
docker pull jupyter/pyspark-notebook

# Check spark and hadoop versions: https://github.com/jupyter/docker-stacks/blob/master/pyspark-notebook/Dockerfile#L11

# Don't add anything to the directory at first or you'll get permission errors:
# https://github.com/jupyter/docker-stacks/issues/542
mkdir ~/data
mkdir ~/spark_dependencies

# Now add the gcs credentials to the mounted volume folder

# Download the appropriate Cloud Storage connector.
# https://cloud.google.com/dataproc/docs/concepts/connectors/cloud-storage
wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-latest.jar -P ~/spark_dependencies

docker run -it --rm -p 8888:8888 --volume ~:/home/jovyan/work jupyter/pyspark-notebook
```

```python3
# Create a new notebook in the /home/jovyan/work directory. Within the notebook, set up google cloud:
import pyspark
from pyspark.sql import *

# Based on `core-site.xml` in https://github.com/GoogleCloudPlatform/bigdata-interop/blob/master/gcs/INSTALL.md doc
spark = SparkSession.builder \
    .master("local") \
    .appName("big_earth") \
    .config("spark.driver.extraClassPath", "./gcs-connector-hadoop2-latest.jar") \
    .config("fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
    .config("fs.gs.project.id", "big-earth-252219") \
    .config("google.cloud.auth.service.account.enable", "true") \
    .config("google.cloud.auth.service.account.json.keyfile", ".gcs/big-earth-252219-fb2e5c109f78.json") \
    .getOrCreate()
```


## Create local Spark cluster
```
export BASE_IMAGE_NAME=skeller88
infrastructure/scripts/docker/build_run_deploy_docker_image.sh spark infrastructure/spark/ False True
```

Start the spark cluster

`docker-compose -f infrastructure/spark/docker-compose.yml up`

Build dependencies

```
pip install -r ./requirements.txt -t ./pip_modules && jar -cvf pip_modules.jar -C ./pip_modules .
jar -cvf src.jar -C . .
```

Submit the job

```bash
/spark/bin/spark-submit \
--master spark://spark-master:7077 \
--packages 'databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11' \
--py-files spark_dist/spark_submit-0.1-deps.zip,spark_dist/spark_submit-0.1.zip \
spark_driver.py data_engineering.spark_metadata_aggregator.py
```

## Run on google cloud
```
gcloud compute instances create-with-container data-prep \
        --zone=us-west1-b \
        --container-image=$BASE_IMAGE_NAME/tensorflow-notebook \
        --container-mount-disk name=data-prep,mount-path=/data-prep,mode=rw \
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-16 \
        --boot-disk-size=10GB \
        --create-disk name=big-earth-data-prep,mode=rw,size=100GB
```

## Gcloud Spark cluster
Copy initialization scripts
```bash
gsutil cp gs://dataproc-initialization-actions/conda/bootstrap-conda.sh gs://sk_spark_ops
gsutil cp gs://dataproc-initialization-actions/conda/install-conda-env.sh gs://sk_spark_ops
gsutil cp /Users/shanekeller/Documents/big_earth_springboard_project/infrastructure/spark_dependencies/conda_environment.yml gs://sk_spark_ops
gsutil cp /Users/shanekeller/Documents/big_earth_springboard_project/infrastructure/spark_dependencies/create_spark_cluster.sh gs://sk_spark_ops
```


```bash
gcloud dataproc clusters create spark-cluster \
--initialization-actions \
gs://sk_spark_ops/create_spark_cluster.sh \
--num-masters=1 \
--num-workers=2 \
--num-preemptible-workers=2 \
--optional-components=ANACONDA \
--region=us-west1
```

## Submit Dataproc job
```bash
gcloud dataproc jobs submit pyspark data_engineering/spark_metadata_aggregator.py --cluster=spark-cluster --region=us-west1
```


# Model training

## Add data to local disk
mkdir ~/jupyter_notebook_files/metadata
mkdir ~/jupyter_notebook_files/raw_rgb
mkdir ~/jupyter_notebook_files/raw_rgb/tiff
gsutil cp -R gs://big_earth/raw_rgb/tiff/S2A_MSIL2A_20170613T101031_0_45 ~/jupyter_notebook_files/raw_rgb/tiff
gsutil cp gs://big_earth/metadata/metadata_01.csv ~/jupyter_notebook_files/metadata
gsutil cp -R gs://big_earth/metadata ~/jupyter_notebook_files

## Run local tensorflow notebook
```bash
docker pull jupyter/tensorflow-notebook

# Don't add anything to the directory at first or you'll get permission errors:
# https://github.com/jupyter/docker-stacks/issues/542
mkdir ~/jupyter_notebook_files

docker run -it --rm -p 8888:8888 --volume ~/jupyter_notebook_files:/home/jovyan/work jupyter/tensorflow-notebook
```


## Set up Google Cloud notebook
Deploy image
`infrastructure/scripts/docker/build_run_deploy_docker_image.sh tensorflow-notebook data_science/jupyter_notebook/ False True`

Deploy notebook
```
# Notebook parameters
# export INPUT_NOTEBOOK_PATH="gs://my-bucket/input.ipynb"
# export OUTPUT_NOTEBOOK_PATH="gs://my-bucket/output.ipynb"
# export PARAMETERS_FILE="params.yaml" # Optional
# export PARAMETERS="-p batch_size 128 -p epochs 40"  # Optional
# export STARTUP_SCRIPT="papermill ${INPUT_NOTEBOOK_PATH} ${OUTPUT_NOTEBOOK_PATH} -y ${PARAMETERS_FILE} ${PARAMETERS}"
# export STARTUP_SCRIPT="docker "

gcloud compute instances create notebook \
        --zone=us-west1-b \
        --image-project=$PROJECT_ID \
        --image=$BASE_IMAGE_NAME/tensorflow-notebook
        --accelerator=nvidia-tesla-k80
        --maintenance-policy=TERMINATE \
        --machine-type=n1-standard-4 \
        --boot-disk-size=50GB \
```

Ssh to instance and mount disk
```bash
# confirm the disk is attached to the instance
lsblk
sudo mkdir -p /mnt/ssd-persistent-disk-200gb
sudo mount /dev/sdb /mnt/ssd-persistent-disk-200gb

# If there are permissions issues with accessing the mounted disk, just give everyone all permissions
chmod 777 .

# Back up notebooks
cp /home/jupyter/*.ipynb /mnt/ssd-persistent-disk-200gb/jupyter
```

# Set up Kubernetes cluster
https://zero-to-jupyterhub.readthedocs.io/en/latest/google/step-zero-gcp.html

```bash
export GOOGLE-EMAIL-ACCOUNT=<your-google-cloud-account-email>

gcloud components install kubectl -y

# https://cloud.google.com/compute/docs/machine-types
# 7.5GB memory, 2 vCPUs
gcloud container clusters create \
  --machine-type n1-standard-4 \
  --num-nodes 1 \
  --zone us-west1-b \
  --cluster-version latest \
  dask

gcloud container node-pools create worker-pool \
    --machine-type n1-standard-2 \
    --num-nodes 7 \
    --preemptible \
    --zone us-west1-b \
    --cluster dask


kubectl create clusterrolebinding cluster-admin-binding \
  --clusterrole=cluster-admin \
  --user=skeller88@gmail.com

gcloud container clusters get-credentials dask --zone us-west1-b --project big-earth-252219

```

set up helm: https://zero-to-jupyterhub.readthedocs.io/en/latest/setup-helm.html

```bash
kubectl --namespace kube-system create serviceaccount tiller
kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller
helm init --service-account tiller --wait
kubectl patch deployment tiller-deploy --namespace=kube-system --type=json --patch='[{"op": "add", "path": "/spec/template/spec/containers/0/command", "value": ["/tiller", "--listen=localhost:44134"]}]'

helm repo update
helm install -f helm_dask_chart.yaml stable/dask

# custom repo
helm install -f helm_dask_chart.yaml /Users/shanekeller/Documents/charts/stable/dask

gcloud container clusters get-credentials dask --zone us-west1-b --project big-earth-252219 \
 && kubectl port-forward $(kubectl get pod --selector="app=dask,component=jupyter,release=mollified-gerbil" --output jsonpath='{.items[0].metadata.name}') 8080:8888
```

To resize
```
 gcloud container clusters resize dask --node-pool worker-pool --num-nodes 5
```