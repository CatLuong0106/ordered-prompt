#!/bin/bash
pip3 install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

nvidia-smi