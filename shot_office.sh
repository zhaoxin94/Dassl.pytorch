#!/bin/bash

DATA="/home/zhao/data/DA"
source_domain="amazon"
target_domain="dslr"
task=$source_domain"2"$target_domain
output_dir="output/shot/office31/"$task
init_weights="output/shot_source/office31/"$task"/model/model.pth.tar-30"
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer SHOT \
--source-domains $source_domain \
--target-domains $target_domain \
--dataset-config-file configs/datasets/da/office31.yaml \
--config-file configs/trainers/da/shot/office31.yaml \
--init-weights $init_weights \
--output-dir $output_dir
