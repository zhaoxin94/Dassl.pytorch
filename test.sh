#!/bin/bash

domain_list="art clipart product real_world"
for (( i=0; i<1; i++ ))
do
    for source_domain in $domain_list
    do
        for target_domain in $domain_list
        do
            if [ $source_domain != $target_domain ]
            then
                output="$source_domain""2""$target_domain"
                echo $output
            fi
        done
    done
done
