#!/bin/bash
export TOXIC_DIR=/scratch/sbp354/DSGA1012/Final_Project/data
export TRAIN_DATASET=founta
export MODE=random_005

python /scratch/pg2255/nlu/Toxic_Debias/src/shallow_subsample.py \
  --overwrite_output_dir \
  --data_dir $TOXIC_DIR \
  --train_dataset $TRAIN_DATASET \
  --output_dir $TOXIC_DIR
  --mode $MODE
  
