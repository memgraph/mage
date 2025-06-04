#!/bin/bash

# this is the tag of the image to get the filesystem from
DOCKER_TAG=$1

# this director will contain / of the container filesystem
OUTPUT_DIR=$2

# create the container without running it
docker create --name temp-container "$DOCKER_TAG"

# export file system
docker export temp-container -o rootfs.tar

# remove temporary container
docker rm temp-container

# extract to output directory
mkdir -p "$OUTPUT_DIR"
tar -xvf rootfs.tar -C "$OUTPUT_DIR"
rm -v rootfs.tar
