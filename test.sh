#!/bin/bash

# DATA="/home/zhao/data/DA"

# CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root $DATA \
# --trainer SourceOnly \
# --source-domains webcam \
# --target-domains dslr \
# --dataset-config-file configs/datasets/da/office_caltech.yaml \
# --config-file configs/trainers/da/source_only/office_caltech.yaml \
# --output-dir output/source_only/office_caltech_test/webcam2dslr \
# --eval-only \
# --model-dir output/source_only/office_caltech/webcam \
# --load-epoch 30


DATA="/home/zhao/data/DA"
dataset=$1
dir_suffix=$2

if [ $dataset = "office_caltech" ]
then
    domain_list="amazon webcam dslr caltech"
elif [ $dataset = "office_home" ]
then    
    domain_list="art clipart product real_world"
elif [ $dataset = "domainnet" ]
then    
    domain_list="clipart infograph painting quickdraw real sketch"
elif [ $dataset = "digit5" ]
then    
    domain_list="mnist mnist_m svhn syn usps"
fi

# source_only model test
for source_domain in $domain_list
do
    target_domain=$source_domain
       
    dataset_cfg="configs/datasets/da/"$dataset".yaml"
    cfg="configs/trainers/da/source_only/"$dataset".yaml"

    task=$source_domain"2"$target_domain
    output_dir="output/source_only/"$dataset"_test/"$dir_suffix"/"$task
    model_dir="output/source_only/"$dataset"_train/"$dir_suffix"/"$source_domain

    CUDA_VISIBLE_DEVICES=1 python tools/train.py \
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
done
