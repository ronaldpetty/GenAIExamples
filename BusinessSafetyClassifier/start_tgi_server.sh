#!/bin/bash

model=mistralai/Mixtral-8x7B-Instruct-v0.1
volume=/media/SSD8T/minminho/huggingface/transformers/ # share a volume with the Docker container to avoid downloading weights every run

sudo docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id $model