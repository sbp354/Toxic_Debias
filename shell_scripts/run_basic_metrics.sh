#!/bin/bash
export TOXIC_DIR=/scratch/sbp354/DSGA1012/Final_Project/models/founta/roberta-large
export FILE=finetune_founta_challenge_covert_comments_results.csv
export LABEL=true_labels
export PRED=predictions
export SCORE=scores

python /scratch/dmm9812/Toxic_Debias/src/shallow_subsample.py \
  --data_dir $TOXIC_DIR \
  --results_csv $TRAIN_DATASET \
  --label_name $LABEL \
  --pred_name $PRED \
  --score_name $SCORE
