#!/bin/bash

DATA="/home/zhao/data/DA"
dataset=$1
dir_suffix=$2

if [ $dataset = "office_caltech" ]
then
    domain_list="amazon webcam dslr caltech"
elif [ $dataset = "" ]
then    
    domain_list=""
fi

for source_domain in $domain_list
do
    for target_domain in $domain_list
    do
        if [ $source_domain != $target_domain ]
        then
            dataset_cfg="configs/datasets/da/"$dataset".yaml"
            cfg="configs/trainers/da/source_only/"$dataset".yaml"

            task=$source_domain"2"$target_domain
            output_dir="output/source_only/"$dataset"_test/"$dir_suffix"/"$task
            model_dir="output/source_only/"$dataset"_train/"$dir_suffix"/"$source_domain

            CUDA_VISIBLE_DEVICES=0 python tools/train.py \
            --root $DATA \
            --trainer SourceOnly \
            --source-domains $source_domain \
            --target-domains $target_domain \
            --dataset-config-file $dataset_cfg \
            --config-file $cfg \
            --output-dir $output_dir \
            --eval-only \
            --model-dir $model_dir \
            --load-epoch 30
        fi
    done
done
