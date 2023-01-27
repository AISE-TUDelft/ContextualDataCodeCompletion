#!/bin/bash

# modify this file as you see fit
# it will use wandb, and create a new sweep by default
# continuing a sweep can be done by providing a sweep id with --sweep_id

python ./finetune_predict.py \
	--hyperparameter_tuning \
  --wandb_training \
  --no_checkpoint_loading \
	--do_train \
	--data_dir /path/to/unixcoder/normal-all \
	--output_dir /path/to/unixcoder/normal-all/output \
	--langs python \
	--overwrite_output_dir \
	--log_dir /path/to/unixcoder/normal-all/output \
  --model_type=gpt2 \
  --pretrain_dir=gpt2 \
  --not_pretrain \
  --save_steps 999999999

# save steps is set to infinity as the code will make checkpoints every epoch
# --langs python only ensures that <EOS> is used instead of { and/or ;, this does not mean it only works with python