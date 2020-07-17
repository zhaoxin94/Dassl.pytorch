#!/bin/bash

DATA="/home/zhao/data/DA"
source_domain="amazon"
target_domain="dslr"
task=$source_domain"2"$target_domain
output_dir="output/shot_source/office31/"$task

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer SourceOnlyShot \
--source-domains $source_domain \
--target-domains $target_domain \
--dataset-config-file configs/datasets/da/office31.yaml \
--config-file configs/trainers/da/source_only_shot/office31.yaml \
--output-dir $output_dir

