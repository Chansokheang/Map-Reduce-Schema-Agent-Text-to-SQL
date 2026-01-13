#!/bin/bash

curl -L -o dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip

# Unzip folders
unzip dev.zip
rm dev.zip
mv dev_20240627 ../data/bird_data
unzip ../data/bird_data/dev_databases.zip -d ../data/bird_data