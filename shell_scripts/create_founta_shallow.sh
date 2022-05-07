#!/bin/bash
export TOXIC_DIR=/scratch/sbp354/DSGA1012/Final_Project/data/civil_comments_0.5
export TRAIN_DATASET=civil_comments_0.5
export MODE=random
export PERCENT=.01

python /scratch/sbp354/DSGA1012/Final_Project/git/Toxic_Debias/src/shallow_subsample.py \
  --overwrite_output_dir \
  --data_dir $TOXIC_DIR \
  --train_dataset $TRAIN_DATASET \
  --output_dir $TOXIC_DIR \
  --mode $MODE \
  --sample_percent $PERCENT