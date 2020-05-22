#!/bin/bash

DATA="/home/zhao/data/DA"

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
--root $DATA \
--trainer SourceOnly \
--source-domains amazon \
--target-domains webcam \
--dataset-config-file configs/datasets/da/office31.yaml \
--config-file configs/trainers/da/source_only/office31.yaml \
--output-dir output/source_only/office31/a2w