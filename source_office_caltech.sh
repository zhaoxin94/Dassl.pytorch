#!/bin/bash

DATA="/home/zhao/data/DA"
source_domain='webcam'
target_domain='dslr'
output_dir="output/source_only/office_caltech/"$source_domain

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
                --root $DATA \
                --trainer SourceOnly \
                --source-domains $source_domain \
                --target-domains $target_domain \
                --dataset-config-file configs/datasets/da/office_caltech.yaml \
                --config-file configs/trainers/da/source_only/office_caltech.yaml \
                --output-dir $output_dir