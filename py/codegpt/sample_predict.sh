#!/bin/bash

# modify this file as you see fit
# this currently runs prediction for the `normal-all` dataset (i.e. TS704-OT with all comments)

python ./finetune_predict.py \
    --eval_line \
    --data_dir /path/to/codegpt/normal-all \
    --output_dir /path/to/codegpt/normal-all/output \
    --checkpoint_path /path/to/codegpt/normal-all/output/${checkpointFolderAbs} \
    --langs python \
    --overwrite_output_dir \
    --log_dir /path/to/codegpt/normal-all/output \
    --model_type=gpt2 \
    --pretrain_dir=gpt2 \
    --not_pretrain