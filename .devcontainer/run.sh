#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker could not be found. Please install Docker."
    exit
fi

# Check for GPU flag
USE_GPU=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) USE_GPU=true ;;
        *) ARGS+="$1 " ;;  # Collect other arguments for Python
    esac
    shift
done

# Construct Docker run command with GPU support if flag is set
if [ "$USE_GPU" = true ]; then
    docker run -it --rm --gpus all python:3.10 python $ARGS
else
    docker run -it --rm python:3.10 python $ARGS
fi

