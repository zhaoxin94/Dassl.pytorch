#!/bin/bash

DATA="/home/zhao/data/DA"

# Ar --> Cl
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer SourceOnlyShot \
--source-domains art \
--target-domains clipart \
--dataset-config-file configs/datasets/da/office31.yaml \
--config-file configs/trainers/da/source_only_shot/office_home.yaml \
--output-dir output/shot_source/office_home/ar2cl

# Ar --> Pr
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer SourceOnlyShot \
--source-domains art \
--target-domains clipart \
--dataset-config-file configs/datasets/da/office31.yaml \
--config-file configs/trainers/da/source_only_shot/office_home.yaml \
--output-dir output/shot_source/office_home/ar2cl