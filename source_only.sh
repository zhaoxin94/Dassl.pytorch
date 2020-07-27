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

# source_only model training
for source_domain in $domain_list
do
    target_domain=$source_domain
    dataset_cfg="configs/datasets/da/"$dataset".yaml"
    cfg="configs/trainers/da/source_only/"$dataset".yaml"
    output_dir="output/source_only/"$dataset"_train/"$dir_suffix"/"$source_domain

    CUDA_VISIBLE_DEVICES=1 python tools/train.py \
    --root $DATA \
    --trainer SourceOnly \
    --source-domains $source_domain \
    --target-domains $target_domain \
    --dataset-config-file $dataset_cfg \
    --config-file $cfg \
    --output-dir $output_dir
done

# source_only model test
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
            --load-epoch 40
        fi
    done
done

# muti-source free domain adaptation
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

    CUDA_VISIBLE_DEVICES=1 python tools/train.py \
    --root $DATA \
    --trainer MSFDA \
    --source-domains $source_domains \
    --target-domains $target_domain \
    --dataset-config-file $dataset_cfg \
    --config-file $cfg \
    --output-dir $output_dir \
    --eval-only \
    --model-dir $model_dir \
    --load-epoch 40
done