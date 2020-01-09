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

# Spark

## Run local standalone spark notebook
```bash
docker pull jupyter/pyspark-notebook

# Check spark and hadoop versions: https://github.com/jupyter/docker-stacks/blob/master/pyspark-notebook/Dockerfile#L11

# Don't add anything to the directory at first or you'll get permission errors:
# https://github.com/jupyter/docker-stacks/issues/542
mkdir ~/jupyter_notebook_files

docker run -it --rm -p 8889:8889 --volume ~/jupyter_notebook_files:/home/jovyan/work jupyter/pyspark-notebook
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

```bash
# Now add the gcs credentials to the mounted volume folder
cp -R ~/.gcs ~/jupyter_notebook_files

# Download the appropriate Cloud Storage connector. 
# https://cloud.google.com/dataproc/docs/concepts/connectors/cloud-storage
wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-latest.jar -P ~/jupyter_notebook_files
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

# Data exploration and preparation

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