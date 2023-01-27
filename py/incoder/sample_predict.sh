#!/bin/bash

# modify this file as you see fit
# this currently runs prediction for the `normal-all` dataset (i.e. TS704-OT with all comments)

python -u ./predict.py \
  --data_path /path/to/datasets/for/unixcoder/normal-all/test.json \
  --output_folder_path /path/to/output/for/unixcoder/normal-all/output