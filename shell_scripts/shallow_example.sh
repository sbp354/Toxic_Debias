#!/bin/bash
export TOXIC_DIR=../Final_Project/data
export TASK_NAME=Shallow
export TRAIN_DATASET=founta/founta_train_shallow_random_0.005.csv
export DEV_DATASET=founta/founta_train_shallow_remainder_random_0.005.csv

export MODEL_DIR=../Final_Project/models/founta

python ../Toxic_Debias/run_toxic.py \
  --model_type roberta \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --save_steps 1000 \
  --logging_steps 1000 \
  --overwrite_output_dir \
  --overwrite_cache \
  --data_dir $TOXIC_DIR \
  --train_dataset $TRAIN_DATASET \
  --dev_dataset $DEV_DATASET \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 5.0 \
  --output_dir $MODEL_DIR