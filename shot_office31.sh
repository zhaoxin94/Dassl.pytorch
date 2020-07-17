#!/bin/bash

DATA="/home/zhao/data/DA"
domain_list="amazon webcam dslr"

for source_domain in $domain_list
do
    for target_domain in $domain_list
    do
        if [ $source_domain != $target_domain ]
        then
            task=$source_domain"2"$target_domain
            output_dir="output/shot/office31/"$task
            init_weights="output/shot_source/office31/"$task"/model/model.pth.tar-30"
            for (( i=0; i<2; i++ ))
            do
                CUDA_VISIBLE_DEVICES=0 python tools/train.py \
                --root $DATA \
                --trainer SHOT \
                --source-domains $source_domain \
                --target-domains $target_domain \
                --dataset-config-file configs/datasets/da/office31.yaml \
                --config-file configs/trainers/da/shot/office31.yaml \
                --init-weights $init_weights \
                --output-dir $output_dir  & sleep 1

                CUDA_VISIBLE_DEVICES=1 python tools/train.py \
                --root $DATA \
                --trainer SHOT \
                --source-domains $source_domain \
                --target-domains $target_domain \
                --dataset-config-file configs/datasets/da/office31.yaml \
                --config-file configs/trainers/da/shot/office31.yaml \
                --init-weights $init_weights \
                --output-dir $output_dir

                wait
            done
        fi
    done
done
