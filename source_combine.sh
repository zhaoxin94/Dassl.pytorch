#!/bin/bash

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

# muti-source 
for target_domain in $domain_list
do
    output_dir="output/source_combine/"$dataset"_test/"$dir_suffix"/"$target_domain
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

    CUDA_VISIBLE_DEVICES=1 python tools/train.py \
    --root $DATA \
    --trainer SourceOnly \
    --source-domains $source_domains \
    --target-domains $target_domain \
    --dataset-config-file $dataset_cfg \
    --config-file $cfg \
    --output-dir $output_dir
done