#!/bin/bash

set -e  # Exit on any error
set -o pipefail  # Fail the pipeline if any command fails

# Load environment variables from .env file
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
fi

# Set default values if not provided
MAGE_VERSION=${MAGE_VERSION:-"3.0.0"}
MAGE_BUILD_ARCH=${MAGE_BUILD_ARCH:-"amd"}
MAGE_BUILD_SCOPE=${MAGE_BUILD_SCOPE:-"with ML"}
MAGE_BUILD_TARGET=${MAGE_BUILD_TARGET:-"prod"}
MAGE_BUILD_TYPE=${MAGE_BUILD_TYPE:-"Release"}
MEMGRAPH_VERSION=${MEMGRAPH_VERSION:-"3.0.0"}
MEMGRAPH_DOWNLOAD_LINK=${MEMGRAPH_DOWNLOAD_LINK:-""}
PUSH_TO_DOCKERHUB=${PUSH_TO_DOCKERHUB:-false}
PUSH_TO_S3=${PUSH_TO_S3:-false}

# Set architecture-based OS variable
if [[ "$MAGE_BUILD_ARCH" == "arm" ]]; then
    OS="ubuntu-24.04-aarch64"
else
    OS="ubuntu-24.04"
fi

# for testing only - use specific deb files for 3.1.0
if [[ "$MAGE_BUILD_ARCH" == "arm" ]]; then
    MEMGRAPH_DOWNLOAD_LINK="https://s3.eu-west-1.amazonaws.com/deps.memgraph.io/memgraph-unofficial/memgraph_3.1.0-rc2_arm64.deb"
else
    MEMGRAPH_DOWNLOAD_LINK="https://s3.eu-west-1.amazonaws.com/deps.memgraph.io/memgraph-unofficial/memgraph_3.1.0-rc2_amd64.deb"
fi


DOCKER_REPOSITORY_NAME="memgraph/memgraph-mage"

# Print selected values for debugging
echo "Mage Version: $MAGE_VERSION"
echo "Architecture: $MAGE_BUILD_ARCH"
echo "Build Scope: $MAGE_BUILD_SCOPE"
echo "Build Target: $MAGE_BUILD_TARGET"
echo "Build Type: $MAGE_BUILD_TYPE"
echo "Memgraph Version: $MEMGRAPH_VERSION"
echo "Memgraph Download Link: $MEMGRAPH_DOWNLOAD_LINK"
echo "Push to DockerHub: $PUSH_TO_DOCKERHUB"
echo "Push to S3: $PUSH_TO_S3"
echo "Using OS: $OS"

# Check if Memgraph download link is valid
if [[ -z "$MEMGRAPH_DOWNLOAD_LINK" ]]; then
    MEMGRAPH_DOWNLOAD_LINK="https://download.memgraph.com/memgraph/v${MEMGRAPH_VERSION}/${OS}/memgraph_${MEMGRAPH_VERSION}-1_${MAGE_BUILD_ARCH}64.deb"
    echo "Using official Memgraph download link: $MEMGRAPH_DOWNLOAD_LINK"
fi

if curl --output /dev/null --silent --head --fail "$MEMGRAPH_DOWNLOAD_LINK"; then
    echo "Memgraph download link is valid"
else
    echo "Memgraph download link is NOT valid"
    exit 1
fi

# Download Memgraph binary
echo "Downloading Memgraph binary..."
curl -L "$MEMGRAPH_DOWNLOAD_LINK" -o "memgraph-${MAGE_BUILD_ARCH}64.deb"

# Set build arguments
BUILD_TARGET="$MAGE_BUILD_TARGET"
BUILD_SCOPE="$MAGE_BUILD_SCOPE"
BUILD_TYPE="$MAGE_BUILD_TYPE"

# Apply special rules for "dev" builds
if [[ "$BUILD_TARGET" == "dev" ]]; then
    BUILD_SCOPE="with ML"
    BUILD_TYPE="RelWithDebInfo"
fi

# Set image tag
IMAGE_TAG="${MAGE_VERSION}-memgraph-${MEMGRAPH_VERSION}"
ARTIFACT_NAME="mage-${IMAGE_TAG}"

if [[ "$MAGE_BUILD_ARCH" == "arm" ]]; then
    ARTIFACT_NAME="${ARTIFACT_NAME}-arm"
fi
if [[ "$BUILD_SCOPE" == "without ML" ]]; then
    ARTIFACT_NAME="${ARTIFACT_NAME}-no-ml"
    IMAGE_TAG="${IMAGE_TAG}-no-ml"
fi
if [[ "$BUILD_TARGET" == "dev" ]]; then
    ARTIFACT_NAME="${ARTIFACT_NAME}-dev"
    IMAGE_TAG="${IMAGE_TAG}-dev"
fi
if [[ "$BUILD_TYPE" == "RelWithDebInfo" ]] && [[ "$BUILD_TARGET" != "dev" ]]; then
    ARTIFACT_NAME="${ARTIFACT_NAME}-relwithdebinfo"
    IMAGE_TAG="${IMAGE_TAG}-relwithdebinfo"
fi

echo "Final Image Tag: $IMAGE_TAG"
echo "Artifact Name: $ARTIFACT_NAME"

# Determine the Dockerfile to use
DOCKERFILE="Dockerfile.v6mgbuild"
if [[ "$BUILD_SCOPE" == "without ML" ]]; then
    DOCKERFILE="Dockerfile.no_ML"
fi

if [[ "$MAGE_BUILD_ARCH" == "arm" ]]; then
    MGBUILD_IMAGE="memgraph/mgbuild:v6_ubuntu-22.04-aarch64"
else
    MGBUILD_IMAGE="memgraph/mgbuild:v6_ubuntu-22.04"
fi

echo "Using Dockerfile: $DOCKERFILE"

# Set up Docker Buildx
docker buildx create --use || true

# Build the Docker image
echo "Building Docker image..."
docker buildx build \
    --target "$BUILD_TARGET" \
    --platform linux/"${MAGE_BUILD_ARCH}64" \
    --tag "$DOCKER_REPOSITORY_NAME:$IMAGE_TAG" \
    --file "$DOCKERFILE" \
    --build-arg BUILD_TYPE="$BUILD_TYPE" \
    --build-arg MGBUILD_IMAGE="$MGBUILD_IMAGE" \
    --load .

# Save Docker image
echo "Saving Docker image..."
mkdir -p output
docker save "$DOCKER_REPOSITORY_NAME:$IMAGE_TAG" > "output/${ARTIFACT_NAME}.tar.gz"

# Push to DockerHub
if [[ "$PUSH_TO_DOCKERHUB" == "true" ]]; then
    echo "Pushing Docker image to DockerHub..."
    docker push "$DOCKER_REPOSITORY_NAME:$IMAGE_TAG"
fi

# Push to S3
if [[ "$PUSH_TO_S3" == "true" ]]; then
    echo "Uploading image to S3..."
    aws s3 cp "output/${ARTIFACT_NAME}.tar.gz" "s3://${S3_DEST_BUCKET}/${S3_DEST_DIR}/${ARTIFACT_NAME}.tar.gz"
fi

echo "Workflow test completed!"
