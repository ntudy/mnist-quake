#!/usr/bin/env bash

image="7394d331bd79"
cur_dir=$(cd `dirname $0` | pwd)
share_mem_size=16G
# 用户根据情况选择使用当前机器的哪些卡进行测试
gpu_device_id="1"
sudo NV_GPU=${gpu_device_id} nvidia-docker run --shm-size=${share_mem_size}  -it --rm -v ${cur_dir} ${image} \	
   /bin/bash local_run.sh


