#!/bin/bash

# modify this file as you see fit
# it will use wandb, and create a new sweep by default
# continuing a sweep can be done by providing a sweep id with --sweep_id

# wandb_training_dataset_dir will automatically be appended with a dataset name (e.g. normal-all, untyped-single_line, explicit-multi_line)
# so make sure to provide the base path

python ./finetune_predict.py \
  --no_checkpoint_loading \
  --wandb_training \
  --wandb_training_dataset_dir /path/to/codegpt \
  --epoch_checkpoints \
  --do_train \
  --data_dir /path/to/codegpt \
  --output_dir /path/to/codegpt \
  --langs python \
  --overwrite_output_dir \
  --log_dir /path/to/codegpt/ \
  --model_type=gpt2 \
  --pretrain_dir=gpt2 \
  --not_pretrain \
  --save_steps 999999999

# save steps is set to infinity as the code will make checkpoints every epoch
# --langs python only ensures that <EOS> is used instead of { and/or ;, this does not mean it only works with python
