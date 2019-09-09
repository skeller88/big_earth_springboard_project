#!/usr/bin/env bash
# Assumes $BASE_IMAGE_NAME has been set
# $1 = docker image name
# $2 = directory path of Dockerfile and env.list
# $3 = if "True", run container locally
# $4 = if "True", deploy container to repository
# $5 = if "True", run Docker image as daemon

export IMAGE_NAME=$BASE_IMAGE_NAME/$1
export FILEDIR=$2
export SHOULD_RUN_DOCKER_IMAGE=$3
export SHOULD_DEPLOY_DOCKER_IMAGE=$4
export SHOULD_RUN_DOCKER_IMAGE_AS_DAEMON=$5

docker build -t $IMAGE_NAME --file $FILEDIR/Dockerfile --build-arg filedir=$FILEDIR .

if [ $SHOULD_RUN_DOCKER_IMAGE  == "True" ]
then
if [ $SHOULD_RUN_DOCKER_IMAGE_AS_DAEMON  == "True" ]
then
docker container run -d --env-file $FILEDIR/env.list -p 5000:5000 $IMAGE_NAME
else
docker container run --env-file $FILEDIR/env.list -p 5000:5000 $IMAGE_NAME
fi
fi

if [ $SHOULD_DEPLOY_DOCKER_IMAGE  == "True" ]
then
docker push $IMAGE_NAME
fi
