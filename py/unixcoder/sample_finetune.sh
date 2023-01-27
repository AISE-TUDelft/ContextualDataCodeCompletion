#!/bin/bash

# modify this file as you see fit

python ./finetune_predict.py \
  --do_train \
  --do_eval \
  --output_dir /your/output/directory \
  --model_name_or_path microsoft/unixcoder-base \
  --max_source_length 936 \
  --max_target_length 64 \
  --beam_size 5