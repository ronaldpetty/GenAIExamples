#!/bin/bash

volume=/home/minminho/workspace

# docker run -it -v $volume:/home/user/workspace annotation-gaudi:latest

docker run -it --name megatron-habana --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -v $volume:/home/user/workspace --cap-add=sys_nice --net=host --ipc=host annotation-gaudi:latest
