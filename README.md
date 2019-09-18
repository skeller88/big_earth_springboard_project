# Local Python Environment
```bash
export PROJECT_DIR=<project-dir>
cd $PROJECT_DIR
python3 -m venv venv
pip install -r requirements.txt

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

# Local Spark Environment
```bash
docker pull jupyter/pyspark-notebook

# Check spark and hadoop versions: https://github.com/jupyter/docker-stacks/blob/master/pyspark-notebook/Dockerfile#L11

# Don't add anything to the directory at first or you'll get permission errors:
# https://github.com/jupyter/docker-stacks/issues/542
mkdir ~/jupyter_notebook_files

docker run -it --rm -p 8888:8888 --volume ~/jupyter_notebook_files:/home/jovyan/work jupyter/pyspark-notebook
```

```python3
# Create a new notebook in the /home/jovyan/work directory. Within the notebook, set up google cloud:
import pyspark
from pyspark.sql import *

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



# Data preparation
Download BigEarth data

```bash
export DIR_WITHOUT_CLOUDS_AND_SNOW=<put BigEarth data without clouds and snow here>

```

```bash
python $PROJECT_DIR/eliminate_snowy_cloudy_patches.py -r ~/Documents/BigEarthNet-v1.0/ -e \
patches_with_cloud_and_shadow.csv patches_with_seasonal_snow.csv -d DIR_WITHOUT_CLOUDS_AND_SNOW
```

