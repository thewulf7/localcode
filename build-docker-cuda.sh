#!/bin/bash

# Exit on any error
set -e

echo "Starting local docker build for CUDA image..."

# Build the docker image
docker build -t ghcr.io/thewulf7/localcode:cuda-latest -f docker/server-cuda/Dockerfile.cuda docker/server-cuda

echo "Build complete! You can now test it with localcode start"
