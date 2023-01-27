#!/bin/bash

# modify this file as you see fit
# it will use wandb, and create a new sweep by default
# continuing a sweep can be done by providing a sweep id with --sweep_id

python ./finetune_predict.py \
	--hyperparameter_tuning \
  --wandb_training \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename /path/to/unixcoder/normal-all/train.txt \
	--dev_filename /path/to/unixcoder/normal-all/dev.json \
	--output_dir /path/to/unixcoder/normal-all/output \
	--max_source_length 936 \
	--max_target_length 64 \
	--beam_size 5