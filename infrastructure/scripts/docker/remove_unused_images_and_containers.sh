#!/usr/bin/env bash

# remove stopped containers
docker ps -aq --no-trun -f status=exited | xargs docker rm

# remove dangling images
docker rmi -f $(docker images --filter "dangling=true" -q --no-trunc)