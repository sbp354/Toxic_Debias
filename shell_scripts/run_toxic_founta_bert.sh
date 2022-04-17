#!/bin/bash
export TOXIC_DIR=/scratch/sbp354/DSGA1012/Final_Project/data
export TASK_NAME=Toxic
export TRAIN_DATASET=founta
export DEV_DATASET=founta

export DATA=$1
export RAN=$2
export MODEL_DIR=/scratch/sbp354/DSGA1012/Final_Project/models/founta

python ../DSGA1012/Final_Project/git/Toxic_Debias/run_toxic.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --save_steps 1000 \
  --logging_steps 1000 \
  --overwrite_output_dir \
  --data_dir $TOXIC_DIR/$DATA \
  --train_dataset $TRAIN_DATASET \
  --dev_dataset $DEV_DATASET \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --output_dir $MODEL_DIR
