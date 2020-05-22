#!/bin/bash

DATA="/home/zhao/data/DA"

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer SHOT \
--source-domains amazon \
--target-domains webcam \
--dataset-config-file configs/datasets/da/office31.yaml \
--config-file configs/trainers/da/shot/office31.yaml \
--output-dir output/shot/office31/a2w