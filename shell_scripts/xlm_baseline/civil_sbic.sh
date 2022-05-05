#!/bin/bash
export TOXIC_DIR=/scratch/sbp354/DSGA1012/Final_Project/data
export TASK_NAME=Toxic
export TRAIN_DATASET=civil_comments_0.5
export DEV_DATASET=SBIC
export MODEL_DIR=/scratch/sbp354/DSGA1012/Final_Project/models/founta/xlm_baseline

python /scratch/dmm9812/Toxic_Debias/run_toxic.py \
  --model_type xlm \
  --model_name_or_path xlm-mlm-en-2048 \
  --task_name $TASK_NAME \
  --no-do_train \
  --do_eval \
  --evaluate_during_training \
  --save_steps 1000 \
  --logging_steps 1000 \
  --overwrite_output_dir \
  --data_dir $TOXIC_DIR \
  --train_dataset $TRAIN_DATASET \
  --dev_dataset $DEV_DATASET \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --output_dir $MODEL_DIR