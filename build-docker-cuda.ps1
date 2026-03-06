$ErrorActionPreference = "Stop"

Write-Host "Starting local docker build for CUDA image..." -ForegroundColor Cyan

# Build the docker image
docker build -t ghcr.io/thewulf7/localcode:cuda-latest -f docker/server-cuda/Dockerfile.cuda docker/server-cuda

Write-Host "Build complete! You can now test it with localcode start" -ForegroundColor Green
