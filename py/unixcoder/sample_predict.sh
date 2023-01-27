#!/bin/bash

# modify this file as you see fit
# this currently runs prediction for the `normal-all` dataset (i.e. TS704-OT with all comments)

python ./finetune_predict.py \
    --do_test \
    --test_filename /path/to/datasets/for/unixcoder/normal-all/test.json \
    --output_dir /path/to/output/for/unixcoder/normal-all/output \
    --model_name_or_path microsoft/unixcoder-base \
    --max_source_length 936 \
    --max_target_length 64 \
    --beam_size 5 \
    --load_model_path /path/to/checkpoints/for/unixcoder/normal-all/pytorch_model.pt