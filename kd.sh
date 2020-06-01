#!/bin/bash

DATA="/home/zhao/data/DA"

CUDA_VISIBLE_DEVICES=1 python tools/train.py \
--root $DATA \
--trainer KD \
--source-domains dslr \
--target-domains amazon \
--dataset-config-file configs/datasets/da/office31.yaml \
--config-file configs/trainers/da/kd/office31.yaml \
--output-dir output/kd/office31/d2a