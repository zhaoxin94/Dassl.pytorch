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

for target_domain in $domain_list
do
    output_dir="output/msfda/"$dataset"_test/"$dir_suffix"/"$target_domain
    source_domains=""
    for source_domain in $domain_list
    do 
        if [ $source_domain != $target_domain ]
        then
            source_domains=$source_domains" "$source_domain
        fi
    done

    dataset_cfg="configs/datasets/da/"$dataset".yaml"
    cfg="configs/trainers/da/source_only/"$dataset".yaml"

    model_dir="output/source_only/"$dataset"_train/"$dir_suffix

    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --root $DATA \
    --trainer MSFDA \
    --source-domains $source_domains \
    --target-domains $target_domain \
    --dataset-config-file $dataset_cfg \
    --config-file $cfg \
    --output-dir $output_dir \
    --eval-only \
    --model-dir $model_dir \
    --load-epoch 30
done




