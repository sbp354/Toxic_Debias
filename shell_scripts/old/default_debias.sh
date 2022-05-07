#!/bin/bash
if [$MODEL = "bert"]
then
  $MODEL_TYPE = bert
  $MODEL_NAME_OR_PATH = bert-base-uncased
elif [$MODEL = "roberta"]
then
  $MODEL_TYPE = roberta
  $MODEL_NAME_OR_PATH = roberta-large
elif [$MODEL = "xlm"]
then
  $MODEL_TYPE = xlm
  $MODEL_NAME_OR_PATH = xlm-mlm-en-2048
else
  exit "Invalid model type {xlm, roberta, bert}"
fi

if [$TRAIN]
then
  if [$EVAL]
  then
    python $RUN_TOXIC_PATH \
      --model_type $MODEL_TYPE \
      --model_name_or_path $MODEL_NAME_OR_PATH \
      --task_name Toxic \
      --do_train \
      --do_eval \
      --evaluate_during_training \
      --save_steps 1000 \
      --logging_steps 1000 \
      --overwrite_output_dir \
      --data_dir $DATA_DIR \
      --train_dataset $TRAIN_DATASET \
      --dev_dataset $DEV_DATASET \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 8 \
      --per_gpu_eval_batch_size 8 \
      --learning_rate 1e-5 \
      --num_train_epochs 3.0 \
      --output_dir $OUTPUT_DIR
  else
    python $RUN_TOXIC_PATH \
      --model_type $MODEL_TYPE \
      --model_name_or_path $MODEL_NAME_OR_PATH \
      --task_name Toxic \
      --do_train \
      --evaluate_during_training \
      --save_steps 1000 \
      --logging_steps 1000 \
      --overwrite_output_dir \
      --data_dir $DATA_DIR \
      --train_dataset $TRAIN_DATASET \
      --dev_dataset $DEV_DATASET \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 8 \
      --per_gpu_eval_batch_size 8 \
      --learning_rate 1e-5 \
      --num_train_epochs 3.0 \
      --output_dir $OUTPUT_DIR
  fi
else
  if [$EVAL]
  then
    python $RUN_TOXIC_PATH \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name Toxic \
    --no-do_train \
    --do_eval \
    --evaluate_during_training \
    --save_steps 1000 \
    --logging_steps 1000 \
    --overwrite_output_dir \
    --data_dir $DATA_DIR \
    --train_dataset $TRAIN_DATASET \
    --dev_dataset $DEV_DATASET \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR
  else
    python $RUN_TOXIC_PATH \
      --model_type $MODEL_TYPE \
      --model_name_or_path $MODEL_NAME_OR_PATH \
      --task_name Toxic \
      --no-do_train \
      --evaluate_during_training \
      --save_steps 1000 \
      --logging_steps 1000 \
      --overwrite_output_dir \
      --data_dir $DATA_DIR \
      --train_dataset $TRAIN_DATASET \
      --dev_dataset $DEV_DATASET \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 8 \
      --per_gpu_eval_batch_size 8 \
      --learning_rate 1e-5 \
      --num_train_epochs 3.0 \
      --output_dir $OUTPUT_DIR
  fi
fi