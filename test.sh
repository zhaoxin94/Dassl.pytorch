#!/bin/bash

DATA="/home/zhao/data/DA"

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer SourceOnly \
--source-domains webcam \
--target-domains dslr \
--dataset-config-file configs/datasets/da/office_caltech.yaml \
--config-file configs/trainers/da/source_only/office_caltech.yaml \
--output-dir output/source_only/office_caltech_test/webcam2dslr \
--eval-only \
--model-dir output/source_only/office_caltech/webcam \
--load-epoch 30