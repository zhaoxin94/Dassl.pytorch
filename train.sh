#!/bin/bash

DATA="/home/zhao/data/DA"
source_domain="infograph"
target_domain="clipart"
task=$source_domain"2"$target_domain
output_dir="output/source_only/domainnet/"$task

CUDA_VISIBLE_DEVICES=1 python tools/train.py \
--root $DATA \
--trainer SourceOnly \
--source-domains $source_domain \
--target-domains $target_domain \
--dataset-config-file configs/datasets/da/domainnet.yaml \
--config-file configs/trainers/da/source_only/domainnet.yaml \
--output-dir $output_dir

